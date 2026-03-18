import asyncio
import json
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import ccxt.async_support as ccxt
from mcp.server.fastmcp import FastMCP
import time

# 添加项目根目录到 sys.path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

# 加载 .env 环境变量
load_dotenv(root_dir / ".env")

from shared.logger import get_logger
from shared.models import MCPErrorResponse
from shared.cache_manager import balance_cache, cache_key_builder
from shared.circuit_breaker import execution_breaker
from asyncache import cached

# 加载 .env 文件，确保能够读取到最新配置
load_dotenv()

logger = get_logger("Execution-Server")
mcp = FastMCP("Execution-Server")

# --- 动态加载风控阈值与安全开关 ---
RISK_CONFIG = {
    "MAX_ORDER_USD": float(os.getenv("MAX_ORDER_USD", 5000)),
    "MAX_POSITION_USD": float(os.getenv("MAX_POSITION_USD", 20000)),
    "MAX_SPREAD_PCT": float(os.getenv("MAX_SPREAD_PCT", 0.005)),
    "VOLATILITY_THRESHOLD": 0.02, # 波动率阈值 (2%)，超过则禁止市价单
    "DRY_RUN_MODE": os.getenv("DRY_RUN_MODE", "True").lower() in ("true", "1", "yes")
}

logger.info(f"Execution-MCP 初始化完成. DRY_RUN_MODE: {RISK_CONFIG['DRY_RUN_MODE']}")

class ExecutionEngine:
    def __init__(self, exchange_id='binanceusdm'):
        self.exchange_id = exchange_id
        self.ex = None
        self.dry_run = str(os.getenv("DRY_RUN_MODE", "True")).lower() == "true"
        # 算法单配置
        self.twap_threshold = float(os.getenv("TWAP_THRESHOLD_USD", 2000))
        self.twap_chunk_size = float(os.getenv("TWAP_CHUNK_USD", 500))
        self.twap_interval_sec = int(os.getenv("TWAP_INTERVAL_SEC", 5))
        self.api_key = os.getenv("BINANCE_API_KEY")
        self.secret = os.getenv("BINANCE_SECRET")
        
        if not self.api_key or not self.secret:
            logger.error("API Key or Secret not found in environment variables!")
            # 可以在这里抛出异常或在 init_exchange 时处理

    async def init_exchange(self):
        if not self.ex:
            if not self.api_key or not self.secret:
                 raise ValueError("Missing API credentials")
                 
            exchange_config = {
                'apiKey': self.api_key,
                'secret': self.secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'} # 默认合约交易
            }
            
            # 动态加载代理
            proxy_url = os.getenv("CCXT_PROXY")
            if proxy_url:
                exchange_config['proxies'] = {
                    'http': proxy_url,
                    'https': proxy_url
                }
                 
            self.ex = getattr(ccxt, self.exchange_id)(exchange_config)
            
            # 如果是 Testnet，开启沙盒模式
            if os.getenv("USE_TESTNET", "False").lower() in ("true", "1", "yes"):
                self.ex.set_sandbox_mode(True)
                
            # 加载市场数据以确保精度计算正确
            await self.ex.load_markets()

    async def validate_risk(self, symbol: str, amount_usd: float, order_type: str = 'market', volatility: float = 0.0):
        """
        硬核风控检查
        """
        # 1. 检查单笔金额
        if amount_usd > RISK_CONFIG["MAX_ORDER_USD"]:
            return False, f"订单金额 ${amount_usd} 超过限额 ${RISK_CONFIG['MAX_ORDER_USD']}"
        
        # 2. 检查波动率限制 (如果波动率过大，禁止市价单)
        if volatility > RISK_CONFIG["VOLATILITY_THRESHOLD"] and order_type == 'market':
             return False, f"当前波动率 ({volatility:.2%}) 过高，禁止执行市价单，请改用限价单 (Limit Order)。"

        # 3. 检查盘口流动性 (防滑点)
        try:
            orderbook = await self.ex.fetch_order_book(symbol, limit=5)
            if not orderbook or not orderbook.get('asks') or not orderbook.get('bids'):
                 return False, "无法获取完整的盘口数据(asks/bids为空)"
                 
            best_ask = orderbook['asks'][0][0]
            best_bid = orderbook['bids'][0][0]
            
            spread = (best_ask - best_bid) / best_bid
            if spread > RISK_CONFIG["MAX_SPREAD_PCT"]:
                return False, f"价差过大 ({spread:.4%}), 超过阈值 {RISK_CONFIG['MAX_SPREAD_PCT']:.2%}, 放弃执行"
                
            return True, "Success"
        except Exception as e:
            return False, f"风控检查时发生错误: {str(e)}"

    async def get_balance(self):
        await self.init_exchange()
        balance = await self.ex.fetch_balance()
        # 假设我们主要关注 USDT
        usdt_balance = balance.get('USDT', {})
        return {
            "total": usdt_balance.get('total', 0.0),
            "free": usdt_balance.get('free', 0.0),
            "used": usdt_balance.get('used', 0.0)
        }
    
    async def cancel_all_orders(self, symbol: str):
        await self.init_exchange()
        await self.ex.cancel_all_orders(symbol)

    async def close_all_positions(self, symbol: str):
        """
        市价平掉指定 symbol 的所有持仓
        """
        await self.init_exchange()
        positions = await self.ex.fetch_positions([symbol])
        
        closed_count = 0
        for position in positions:
            size = float(position['contracts'])
            side = position['side'] # 'long' or 'short'
            
            if size > 0:
                # 平仓方向相反
                close_side = 'sell' if side == 'long' else 'buy'
                await self.ex.create_market_order(symbol, close_side, size, params={'reduceOnly': True})
                closed_count += 1
                
        return closed_count

    async def run_twap_background(self, symbol: str, side: str, total_amount_usd: float):
        """[核心算法] 后台静默执行 TWAP 时间加权算法单"""
        task_id = f"TWAP_{symbol}_{side}_{total_amount_usd}"
        logger.info(f"🚀 启动 TWAP 算法引擎 [{task_id}]: 总额 ${total_amount_usd}, 每刀 ${self.twap_chunk_size}, 间隔 {self.twap_interval_sec}s")
        
        remaining_usd = total_amount_usd
        slice_count = 0

        try:
            await self.init_exchange()
            while remaining_usd > 0:
                slice_count += 1
                current_slice_usd = min(self.twap_chunk_size, remaining_usd)
                remaining_usd -= current_slice_usd

                logger.info(f"⏳ TWAP [{task_id}] 第 {slice_count} 刀: 准备 {side} {symbol} ${current_slice_usd} (剩余未执行: ${remaining_usd})")

                if self.dry_run:
                    logger.info(f"🛡️ [DRY-RUN] TWAP 模拟切片成交: {side} {symbol} ${current_slice_usd}")
                else:
                    # 真实切片市价吃单
                    ticker = await self.ex.fetch_ticker(symbol)
                    price = ticker['last']
                    amount = current_slice_usd / price
                    await self.ex.create_market_order(symbol, side, amount)
                    logger.info(f"✅ TWAP 真实切片成交: {side} {symbol} {amount} @ {price}")

                if remaining_usd > 0:
                    await asyncio.sleep(self.twap_interval_sec)

            logger.info(f"🏁 TWAP 算法单 [{task_id}] 全部隐形执行完毕！")
            # 执行完毕后清空余额缓存
            balance_cache.clear()
        except Exception as e:
            logger.error(f"❌ TWAP [{task_id}] 异常中断: {e}")

engine = ExecutionEngine()

@mcp.tool()
@cached(cache=balance_cache, key=cache_key_builder)
async def get_account_balance() -> str:
    """
    [Agent 工具] 获取账户当前的资金情况 (USDT)。(内置 10s TTL 缓存)
    """
    logger.info("🌐 [Cache Miss] 正在向交易所请求真实账户余额...")
    try:
        await engine.init_exchange()
        balance = await engine.get_balance()
        logger.info(f"查询余额成功: {balance}")
        return str(balance)
    except Exception as e:
        logger.error(f"查询余额失败: {str(e)}")
        return MCPErrorResponse(status="error", error_code="BALANCE_QUERY_FAILED", message=str(e)).model_dump_json()

@mcp.tool()
async def execute_smart_order(symbol: str, side: str, amount_usd: float, order_type: str = 'market', price: float = None, sl_price: float = None, volatility: float = 0.0) -> str:
    """
    [Agent 专用工具] 执行带风控校验的智能下单指令。
    
    Args:
        symbol (str): 交易对，如 'BTC/USDT'。
        side (str): 'buy' 或 'sell'。
        amount_usd (float): 下单金额 (USDT)。
        order_type (str): 'market' (市价) 或 'limit' (限价)。
        price (float, optional): 限价单必须指定价格。
        sl_price (float, optional): 止损触发价格 (若设置，会自动挂止损单)。
        volatility (float, optional): 当前市场波动率 (来自 Indicator-MCP)，用于风控判断。
        
    Returns:
        str: 订单结果 JSON 字符串。
    """
    # 1. 物理层面的第一道门：熔断检查
    allowed, reason = execution_breaker.is_allowed()
    if not allowed:
        logger.error(f"🛡️ 熔断拦截开仓请求: {reason}")
        return MCPErrorResponse(status="rejected", error_code="CIRCUIT_BREAKER_OPEN", message=f"全局熔断已触发: {reason}").model_dump_json()

    try:
        await engine.init_exchange()
        
        # 1. 执行前置风控
        is_safe, reason = await engine.validate_risk(symbol, amount_usd, order_type, volatility)
        if not is_safe:
            logger.warn(f"风控拦截 ({symbol}): {reason}")
            return MCPErrorResponse(status="rejected", error_code="RISK_CHECK_FAILED", message=reason).model_dump_json()

        # 2. 计算下单数量
        ticker = await engine.ex.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # 如果是限价单，且未提供价格，则报错
        if order_type == 'limit' and not price:
             return MCPErrorResponse(status="error", error_code="INVALID_PARAMS", message="Limit order requires a price.").model_dump_json()
        
        exec_price = price if order_type == 'limit' else current_price
        amount = amount_usd / exec_price
        
        # 精度调整 (简单处理，实际应使用 exchange.amount_to_precision)
        # 这里暂且信任 ccxt 的自动处理或手动截断
        # amount = float(engine.ex.amount_to_precision(symbol, amount)) 

        logger.info(f"准备执行: {side.upper()} {symbol} {amount:.4f} @ {order_type} (Est. Price: {exec_price})")

        # ==========================================
        # 智能路由：大额订单自动降级为 TWAP 后台任务
        # ==========================================
        if amount_usd >= engine.twap_threshold:
            # Fire and forget: 将任务抛入后台，不阻塞 Agent
            asyncio.create_task(engine.run_twap_background(symbol, side, amount_usd))
            return json.dumps({
                "status": "success", 
                "message": f"订单金额 ${amount_usd} 触发大额路由，已移交 TWAP 算法引擎后台切片执行。Agent 可继续观望。"
            }, ensure_ascii=False)

        # ---------------------------------------------------------
        # ⚠️ 核心安全拦截 (Dry-Run 模式检查)
        # ---------------------------------------------------------
        if RISK_CONFIG["DRY_RUN_MODE"]:
            logger.warn(f"🛡️ [DRY-RUN 拦截] 模拟下单成功，未发送至交易所: {side.upper()} {symbol} {amount:.4f} @ {exec_price}")
            
            # 主动清空余额缓存，确保下一次查询是最新数据
            balance_cache.clear()
            logger.info("♻️ 订单已执行(Dry-Run)，已主动清空本地资产缓存。")
            
            # 模拟返回成功订单结构
            mock_order_id = f"mock_{int(time.time()*1000)}"
            return str({
                "status": "success_dry_run", 
                "order_id": mock_order_id, 
                "sl_order_id": f"{mock_order_id}_sl" if sl_price else None,
                "filled": amount,
                "average": exec_price,
                "note": "This is a simulated order generated in DRY_RUN_MODE."
            })

        # 3. 发送物理主订单
        if order_type == 'market':
            order = await engine.ex.create_market_order(symbol, side, amount)
        else:
            order = await engine.ex.create_limit_order(symbol, side, amount, exec_price)
            
        logger.info(f"主订单物理下单成功: ID={order['id']}, Status={order['status']}")
        
        # 主动清空余额缓存，确保下一次查询是最新数据
        balance_cache.clear()
        execution_breaker.record_success() # 成功发单，重置熔断器
        logger.info("♻️ 订单已执行，已主动清空本地资产缓存。")
        
        # 4. 同步提交止损单 (Reduce-only)
        sl_order_id = None
        if sl_price:
            try:
                sl_order = await engine.ex.create_order(
                    symbol=symbol,
                    type='STOP_MARKET',
                    side='buy' if side == 'sell' else 'sell',
                    amount=amount,
                    params={
                        'stopPrice': sl_price,
                        'reduceOnly': True
                    }
                )
                sl_order_id = sl_order['id']
                logger.info(f"止损单设置成功: ID={sl_order_id} @ {sl_price}")
            except Exception as e:
                logger.error(f"止损单设置失败 (主单已成): {str(e)}")
                return MCPErrorResponse(status="partial_success", message=f"主单成功但止损失败: {str(e)}", data={"order_id": order['id']}).model_dump_json()

        return str({
            "status": "success",
            "order_id": order['id'],
            "sl_order_id": sl_order_id,
            "filled": order.get('filled', amount),
            "average": order.get('average', exec_price)
        })

    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        # 网络或交易所异常，立刻触发熔断计数
        execution_breaker.record_failure()
        logger.error(f"执行层网络崩溃: {str(e)}")
        return MCPErrorResponse(status="error", error_code="EXCHANGE_NETWORK_ERROR", message=f"交易所连接失败: {str(e)}").model_dump_json()

    except Exception as e:
        logger.error(f"下单执行异常: {str(e)}")
        return MCPErrorResponse(status="error", error_code="EXECUTION_ERROR", message=str(e)).model_dump_json()

@mcp.tool()
async def kill_all_positions(symbol: str) -> str:
    """
    [Agent 工具] 一键清仓撤单 (Kill Switch)。撤销所有挂单并市价平掉所有持仓。
    """
    # Kill Switch 是逃生通道，不应该受普通熔断器限制，除非物理断网
    # 但为了防止死循环报错，也可以加一个简单的判断，或者赋予它最高优先级（绕过熔断）
    # 这里选择绕过熔断，因为这是用来救命的
    
    try:
        await engine.init_exchange()
        logger.warn(f"🚨 收到清仓指令: {symbol}")
        
        # 1. 撤销所有挂单
        await engine.cancel_all_orders(symbol)
        logger.info(f"已撤销 {symbol} 所有挂单。")
        
        # 2. 平掉所有仓位
        closed_count = await engine.close_all_positions(symbol)
        logger.info(f"已平仓 {symbol} 的 {closed_count} 个持仓。")
        
        return str({"status": "success", "message": f"Kill switch executed for {symbol}. Orders cancelled, {closed_count} positions closed."})

    except Exception as e:
        logger.error(f"熔断失败: {str(e)}")
        return MCPErrorResponse(status="error", error_code="KILL_SWITCH_FAILED", message=str(e)).model_dump_json()

if __name__ == "__main__":
    mcp.run()
