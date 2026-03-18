import asyncio
import ccxt.pro as ccxtpro
import os
from shared.logger import get_logger

logger = get_logger("WS-Memory-Pool")

class WsMemoryPool:
    _instance = None

    def __new__(cls):
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
                
            self.ex = ccxtpro.binance(exchange_config)
            
            # 【修复点 1】：显式适配 Testnet (根据您的环境变量)
            if os.getenv("USE_TESTNET", "False").lower() in ("true", "1", "yes"):
                self.ex.set_sandbox_mode(True)
                logger.info("🧪 WS 引擎已开启 Sandbox 测试网模式")

            logger.info("🟢 CCXT Pro WebSocket Client 初始化完成！")

    async def subscribe(self, symbol: str, timeframe: str):
        """建立长连接并使用 REST 填充首帧数据"""
        await self.init_exchange()
        
        if symbol not in self.active_symbols:
            self.active_symbols.add(symbol)
            logger.info(f"🔌 启动混合订阅池 (REST快照+WS长连): {symbol} @ {timeframe}")
            
            # 【核心修复点 2】：REST 首帧快照，瞬间蓄水，解决死等问题！
            try:
                logger.info(f"⏳ 正在拉取 REST 首帧全量快照...")
                ohlcv, ob, ticker, trades = await asyncio.gather(
                    self.ex.fetch_ohlcv(symbol, timeframe, limit=500),
                    self.ex.fetch_order_book(symbol, limit=20),
                    self.ex.fetch_ticker(symbol),
                    self.ex.fetch_trades(symbol, limit=200)
                )
                self.pool["ohlcv"][symbol] = ohlcv
                self.pool["orderbook"][symbol] = ob
                self.pool["ticker"][symbol] = ticker
                self.pool["trades"][symbol] = trades
                logger.info(f"✅ {symbol} 首帧快照蓄水完毕！进入超低延迟模式。")
            except Exception as e:
                logger.error(f"❌ 首帧快照拉取失败: {e}")

            # 启动 WS 守护协程，后续实时覆盖内存池
            self.tasks.append(asyncio.create_task(self._watch_ticker(symbol)))
            self.tasks.append(asyncio.create_task(self._watch_orderbook(symbol)))
            self.tasks.append(asyncio.create_task(self._watch_trades(symbol)))
            self.tasks.append(asyncio.create_task(self._watch_ohlcv(symbol, timeframe)))

    # ================= 后台守护任务 (保持被动覆盖逻辑) =================
    async def _watch_ticker(self, symbol):
        while True:
            try:
                self.pool["ticker"][symbol] = await self.ex.watch_ticker(symbol)
            except Exception:
                await asyncio.sleep(2)

    async def _watch_orderbook(self, symbol):
        while True:
            try:
                self.pool["orderbook"][symbol] = await self.ex.watch_order_book(symbol)
            except Exception:
                await asyncio.sleep(2)

    async def _watch_trades(self, symbol):
        while True:
            try:
                trades = await self.ex.watch_trades(symbol)
                self.pool["trades"][symbol] = trades[-200:] if len(trades) > 200 else trades
            except Exception:
                await asyncio.sleep(2)

    async def _watch_ohlcv(self, symbol, timeframe):
        while True:
            try:
                self.pool["ohlcv"][symbol] = await self.ex.watch_ohlcv(symbol, timeframe)
            except Exception:
                await asyncio.sleep(2)

    # ================= 对外数据接口 =================
    async def get_market_data(self, symbol: str, timeframe: str):
        if symbol not in self.active_symbols:
            await self.subscribe(symbol, timeframe)
            
        return {
            "ohlcv": self.pool["ohlcv"].get(symbol, []),
            "orderbook": self.pool["orderbook"].get(symbol, {'bids': [], 'asks': []}),
            "ticker": self.pool["ticker"].get(symbol, {'last': 0}),
            "trades": self.pool["trades"].get(symbol, [])
        }