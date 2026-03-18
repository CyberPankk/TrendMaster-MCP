import asyncio
import sys
import os
from pathlib import Path
import ccxt.async_support as ccxt
from dotenv import load_dotenv

# 添加项目根目录到 sys.path
root_dir = Path(__file__).parent
sys.path.append(str(root_dir))

# 加载环境变量
load_dotenv(root_dir / ".env")

from shared.logger import get_logger
from servers.indicator_mcp.engines.hmm_engine import HMMEngine
from servers.indicator_mcp.engines.smc_engine import SMCEngine
from servers.indicator_mcp.engines.orderflow_engine import OrderflowEngine
from servers.execution_mcp.server import ExecutionEngine

logger = get_logger("Integration-Test")

async def get_testnet_exchange():
    """获取 Binance Testnet 的 Exchange 实例"""
    api_key = os.getenv("BINANCE_API_KEY")
    secret = os.getenv("BINANCE_SECRET")
    
    exchange = ccxt.binanceusdm({
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}, # 合约交易
    })
    
    # 开启沙盒模式 (Testnet)
    exchange.set_sandbox_mode(True)
    return exchange

async def simulate_agent_decision_loop(symbol: str = "BTC/USDT"):
    logger.info(f"=== 🚀 启动 TrendMaster Testnet 集成测试: {symbol} ===")

    ex = await get_testnet_exchange()
    
    try:
        # ---------------------------------------------------------
        # 1. 感知层 (Perception) - Market-MCP (真实 API)
        # ---------------------------------------------------------
        logger.info("👉 [1/4] 启动感知层 (Market-MCP) - Testnet...")
        
        # 加载市场信息
        await ex.load_markets()
        
        # 获取最新盘口和价格
        ticker = await ex.fetch_ticker(symbol)
        orderbook = await ex.fetch_order_book(symbol, limit=20)
        current_price = ticker['last']
        
        logger.info(f"Market-MCP: {symbol} 当前价格 {current_price}, 24H成交额 {ticker['quoteVolume']}")

        # ---------------------------------------------------------
        # 2. 认知层 (Cognition) - Indicator-MCP (真实计算)
        # ---------------------------------------------------------
        logger.info("👉 [2/4] 启动认知层 (Indicator-MCP)...")
        
        # a. 订单流计算
        of_engine = OrderflowEngine()
        ofi_score = of_engine.calculate_ofi(symbol, orderbook)
        logger.info(f"Indicator-MCP (Orderflow): 盘口买卖失衡值 (OFI) = {ofi_score}")

        # b. 市场状态识别 (HMM) - 需要历史数据
        logger.info("Indicator-MCP (HMM): 正在获取 K 线数据并训练模型...")
        ohlcv = await ex.fetch_ohlcv(symbol, '1h', limit=500)
        if not ohlcv:
            logger.warn("HMM: 无法获取足够的 K 线数据")
            market_regime = "UNKNOWN"
        else:
            import pandas as pd
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            hmm_engine = HMMEngine()
            hmm_result = hmm_engine.train_and_predict(df)
            
            # 简单映射状态
            if hmm_result:
                # 这里简单取状态 ID，实际需要 interpret
                market_regime = f"STATE_{hmm_result['current_state']}"
            else:
                market_regime = "UNKNOWN"
                
        logger.info(f"Indicator-MCP (HMM): 当前市场状态判定为 [{market_regime}]")

        # c. 聪明钱结构识别 (SMC)
        # 同样使用 df
        if 'df' in locals():
            smc_engine = SMCEngine(df)
            smc_analysis = smc_engine.analyze()
            smc_structure = smc_analysis['structure']['structure']
            smc_event = smc_analysis['structure']['last_event']
            logger.info(f"Indicator-MCP (SMC): 结构 [{smc_structure}], 事件 [{smc_event}]")
        else:
            smc_event = "NONE"

        # ---------------------------------------------------------
        # 3. 决策层 (Decision) - 模拟 Agent 逻辑
        # ---------------------------------------------------------
        logger.info("👉 [3/4] 启动决策层 (Agent Reasoning)...")
        
        # 为了测试下单，我们强制给出一个 BUY 信号 (仅用于验证 Execution 通路)
        # 实际逻辑应遵循 Strategy-SOP
        decision = {
            "action": "EXECUTE_TRADE", 
            "side": "buy", 
            "amount_usd": 120, # 满足 Testnet 名义价值大于 100 USDT 的要求
            "reason": "Testnet 集成测试强制开仓验证"
        }
        
        logger.info(f"Agent 最终决策: {decision['action']} | 理由: {decision['reason']}")

        # ---------------------------------------------------------
        # 4. 执行层 (Execution) - Execution-MCP (真实下单)
        # ---------------------------------------------------------
        if decision["action"] == "EXECUTE_TRADE":
            logger.info("👉 [4/4] 启动执行层 (Execution-MCP) - Testnet...")
            
            exec_engine = ExecutionEngine()
            # 注入 Testnet Exchange 实例，覆盖默认的 Mainnet 连接
            exec_engine.ex = ex 
            
            # 风控前置校验
            is_safe, risk_msg = await exec_engine.validate_risk(symbol, decision["amount_usd"])
            
            if not is_safe:
                logger.error(f"Execution-MCP 拒绝执行: {risk_msg}")
            else:
                logger.info("Execution-MCP 风控通过！正在 Testnet 下单...")
                
                # 计算数量
                amount = decision["amount_usd"] / current_price
                
                # 币安合约(BTC/USDT)最小名义价值通常要求 > 5 USDT，有些是 100 USDT
                # 这里我们直接设置一个稍微大一点的固定测试数量
                amount = round(amount, 3) # 保留3位小数
                if amount < 0.002: 
                    amount = 0.002
                    
                # 为了确保 Notional Value > 100 USDT，我们根据当前价格反算
                min_amount_for_100_usd = round((105 / current_price), 3)
                if amount < min_amount_for_100_usd:
                    amount = min_amount_for_100_usd
                    logger.warn(f"调整下单数量至满足名义价值>100的要求: {amount} BTC")

                try:
                    # 真实下单 (Testnet) - 使用 create_order 以兼容更多参数
                    # 确保参数类型正确
                    amount = float(amount)
                    order = await ex.create_order(
                        symbol=symbol, 
                        type='market', 
                        side=decision['side'].lower(), 
                        amount=amount
                    )
                    logger.info(f"✅ [TESTNET] 下单成功: {decision['side']} {symbol} {amount} @ Market")
                    logger.info(f"Order ID: {order['id']}, Status: {order['status']}")
                except Exception as e:
                    logger.error(f"❌ [TESTNET] 下单失败: {str(e)}")

        else:
            logger.info("👉 [4/4] 执行层: 无需操作。")

    except Exception as e:
        logger.error(f"系统集成测试异常: {str(e)}")
        import traceback
        traceback.print_exc()
        
    finally:
        await ex.close()
        logger.info("=== 🏁 Testnet 测试结束 ===")

if __name__ == "__main__":
    asyncio.run(simulate_agent_decision_loop("BTC/USDT"))
