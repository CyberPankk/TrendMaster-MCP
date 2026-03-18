import asyncio
import ccxt.pro as ccxtpro
import os
from shared.logger import get_logger

logger = get_logger("WS-Memory-Pool")

class WsMemoryPool:
    _instance = None

    def __new__(cls):
        """单例模式：确保全局只有一个 WebSocket 连接池"""
        if cls._instance is None:
            cls._instance = super(WsMemoryPool, cls).__new__(cls)
            cls._instance.ex = None
            cls._instance.pool = {"ticker": {}, "orderbook": {}, "trades": {}, "ohlcv": {}}
            cls._instance.active_symbols = set()
            cls._instance.tasks = []
        return cls._instance

    async def init_exchange(self):
        if self.ex is None:
            exchange_config = {'enableRateLimit': True, 'options': {'defaultType': 'swap'}}
            proxy_url = os.getenv("CCXT_PROXY")
            if proxy_url:
                exchange_config['proxies'] = {'http': proxy_url, 'https': proxy_url}
                
            # 核心升级：使用 ccxtpro (WebSocket 版本)
            self.ex = ccxtpro.binance(exchange_config)
            # 检查是否使用 Testnet
            if os.getenv("USE_TESTNET", "False").lower() in ("true", "1", "yes"):
                 self.ex.set_sandbox_mode(True)
                 
            logger.info("🟢 CCXT Pro WebSocket Client 初始化完成！")

    async def subscribe(self, symbol: str, timeframe: str):
        """建立长连接订阅并开启守护任务"""
        await self.init_exchange()
        
        if symbol not in self.active_symbols:
            self.active_symbols.add(symbol)
            logger.info(f"🔌 建立 WebSocket 并发订阅池: {symbol} @ {timeframe}")
            
            # 启动 4 个后台守护协程，不断向内存池注水
            self.tasks.append(asyncio.create_task(self._watch_ticker(symbol)))
            self.tasks.append(asyncio.create_task(self._watch_orderbook(symbol)))
            self.tasks.append(asyncio.create_task(self._watch_trades(symbol)))
            self.tasks.append(asyncio.create_task(self._watch_ohlcv(symbol, timeframe)))
            
            # 阻塞等待所有通道的初次全量快照完成
            logger.info(f"⏳ 正在拉取首次全量快照 (后续将降级为 0 延迟读取)...")
            while True:
                has_ticker = symbol in self.pool["ticker"]
                has_ob = symbol in self.pool["orderbook"]
                has_trades = symbol in self.pool["trades"]
                has_ohlcv = symbol in self.pool["ohlcv"]
                if has_ticker and has_ob and has_trades and has_ohlcv:
                    logger.info(f"✅ {symbol} 内存池蓄水完毕！进入超低延迟模式。")
                    break
                await asyncio.sleep(0.1)

    # ================= 后台守护任务 =================
    async def _watch_ticker(self, symbol):
        while True:
            try:
                self.pool["ticker"][symbol] = await self.ex.watch_ticker(symbol)
            except Exception as e:
                logger.debug(f"WS Ticker 重连中... {e}")
                await asyncio.sleep(1)

    async def _watch_orderbook(self, symbol):
        while True:
            try:
                self.pool["orderbook"][symbol] = await self.ex.watch_order_book(symbol)
            except Exception as e:
                await asyncio.sleep(1)

    async def _watch_trades(self, symbol):
        while True:
            try:
                trades = await self.ex.watch_trades(symbol)
                # 仅保留最近 200 笔用于 Aggressor 计算，防止内存溢出
                self.pool["trades"][symbol] = trades[-200:] if len(trades) > 200 else trades
            except Exception as e:
                await asyncio.sleep(1)

    async def _watch_ohlcv(self, symbol, timeframe):
        while True:
            try:
                # CCXT Pro 会自动在第一次获取历史数据，之后增量更新
                self.pool["ohlcv"][symbol] = await self.ex.watch_ohlcv(symbol, timeframe)
            except Exception as e:
                await asyncio.sleep(1)

    # ================= 对外数据接口 =================
    async def get_market_data(self, symbol: str, timeframe: str):
        """对外的极速数据读取接口"""
        if symbol not in self.active_symbols:
            await self.subscribe(symbol, timeframe)
        
        # 直接从物理内存中切片读取
        return {
            "ohlcv": self.pool["ohlcv"][symbol],
            "orderbook": self.pool["orderbook"][symbol],
            "ticker": self.pool["ticker"][symbol],
            "trades": self.pool["trades"][symbol]
        }