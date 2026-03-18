import sys
import os
from pathlib import Path
from dotenv import load_dotenv
import time
import pandas as pd
import ccxt.async_support as ccxt
from mcp.server.fastmcp import FastMCP

# 添加项目根目录到 sys.path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

# 加载 .env 环境变量
load_dotenv(root_dir / ".env")

from shared.logger import get_logger
from shared.models import MarketContext, MarketRegime, VolatilityLevel, MCPErrorResponse
from servers.indicator_mcp.engines.smc_engine import SMCEngine
from servers.indicator_mcp.engines.hmm_engine import HMMEngine
from servers.indicator_mcp.engines.orderflow_engine import OrderflowEngine
from servers.indicator_mcp.engines.ws_engine import WsMemoryPool  # 引入新引擎
import numpy as np
import asyncio
import json
import time
from shared.cache_manager import ohlcv_cache, market_cache, cache_key_builder
from asyncache import cached

# 初始化
logger = get_logger("Indicator-Server")
mcp = FastMCP("Indicator-Server")

# 实例化全局内存池
ws_pool = WsMemoryPool()

# 单例引擎实例
smc_engine_cache = {} # SMC 引擎通常是无状态的，或者每次重新实例化，这里仅作占位
hmm_engine = HMMEngine(n_components=3) # HMM 也可以每次实例化，但如果模型很大可以持久化
orderflow_engine = OrderflowEngine()   # Orderflow 必须持久化以计算 OFI 增量

# ==========================================
# 🧠 注入全局记忆流：记录各个资产的宏观状态
# ==========================================
STATE_MEMORY = {}

logger.info("Indicator-MCP Server initializing...")

def get_exchange_config():
    """动态获取带有代理的 CCXT 配置"""
    config = {
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'}
    }
    proxy_url = os.getenv("CCXT_PROXY")
    if proxy_url:
        config['proxies'] = {
            'http': proxy_url,
            'https': proxy_url,
        }
    return config

# @mcp.tool()
async def analyze_smc(symbol: str, timeframe: str = '1h') -> str:
    """
    基于 SMC (Smart Money Concepts) 分析市场结构。
    """
    start_time = time.time()
    logger.info(f"Received SMC analysis request for {symbol} on {timeframe}")
    
    config = get_exchange_config()
    exchange = ccxt.binanceusdm(config)
    if os.getenv("USE_TESTNET", "False").lower() in ("true", "1", "yes"):
        exchange.set_sandbox_mode(True)
    
    try:
        # 1. 获取 K 线数据 (默认最近 100 根足够分析近期结构)
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=100)
        
        if not ohlcv:
            raise ValueError(f"No OHLCV data returned for {symbol}")
            
        # 转换为 DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 2. 运行 SMC 引擎
        engine = SMCEngine(df)
        analysis_result = engine.analyze()
        
        structure_data = analysis_result['structure']
        fvg_data = analysis_result['fvg']
        
        # 3. 映射到 shared.models.MarketContext
        # 将字符串枚举转为 Enum 对象
        regime_map = {
            "BULL": MarketRegime.BULL,
            "BEAR": MarketRegime.BEAR,
            "SIDEWAYS": MarketRegime.SIDEWAYS
        }
        regime = regime_map.get(structure_data['structure'], MarketRegime.SIDEWAYS)
        
        # 简单估算波动率 (ATR 简化版：High - Low 的平均值)
        atr = (df['high'] - df['low']).mean()
        volatility = VolatilityLevel.MEDIUM # 暂时硬编码，后续可接 ATR 逻辑
        
        # 提取支撑阻力位
        support_levels = []
        if structure_data['key_support']:
            support_levels.append(structure_data['key_support'])
            
        resistance_levels = []
        if structure_data['key_resistance']:
            resistance_levels.append(structure_data['key_resistance'])
            
        # 生成脱水 Insight
        insight = f"当前市场结构为 {structure_data['structure']}。"
        if structure_data['last_event'] != "NONE":
            insight += f" 最近发生了 {structure_data['last_event']}。"
            
        if fvg_data['has_fvg']:
            insight += f" 并在 {fvg_data['gap_bottom']:.2f}-{fvg_data['gap_top']:.2f} 存在 {fvg_data['type']} FVG 缺口，"
            if fvg_data['type'] == 'BULLISH':
                insight += "建议在此区间寻找顺势买入机会。"
                support_levels.append(fvg_data['gap_top']) # FVG 上沿作为支撑
            else:
                insight += "建议在此区间寻找顺势卖出机会。"
                resistance_levels.append(fvg_data['gap_bottom']) # FVG 下沿作为阻力
        else:
            insight += " 附近无明显 FVG 缺口，建议观望。"

        # 构造最终 Context
        context = MarketContext(
            symbol=symbol,
            regime=regime,
            volatility=volatility,
            atr=float(atr),
            support_levels=[float(x) for x in support_levels],
            resistance_levels=[float(x) for x in resistance_levels],
            insight=f"[SMC分析] {insight}"
        )
        
        elapsed = time.time() - start_time
        logger.info(f"SMC Analysis completed for {symbol} in {elapsed:.4f}s")
        
        return context.model_dump_json()

    except Exception as e:
        logger.error(f"Error in analyze_smc: {str(e)}")
        return MCPErrorResponse(
            status="error",
            error_code="SMC_ANALYSIS_ERROR",
            message=str(e)
        ).model_dump_json()
        
    finally:
        await exchange.close()

def interpret_hmm_state(means: np.ndarray, covars: np.ndarray, current_state: int) -> tuple[MarketRegime, VolatilityLevel, str]:
    """
    辅助函数：根据 HMM 模型参数解读当前状态含义。
    
    逻辑：
    - 比较各状态的波动率 (range 的均值) 和 收益率 (log_return 的方差)
    - 波动率最高的状态 -> VOLATILE_TREND
    - 波动率最低的状态 -> QUIET_SIDEWAYS
    - 其他 -> REVERSAL_ZONE / NORMAL
    """
    # means 结构: [n_components, n_features] -> 特征顺序 [log_return, range]
    # 我们主要关注第二个特征 'range' (波动率) 的均值来区分状态
    
    volatility_means = means[:, 1] # 获取所有状态的波动率均值
    
    # 排序状态：按波动率从小到大
    sorted_indices = np.argsort(volatility_means)
    
    # 定义状态映射
    state_labels = {}
    
    # 波动率最小的状态 -> 低波动震荡
    state_labels[sorted_indices[0]] = {
        "regime": MarketRegime.SIDEWAYS,
        "volatility": VolatilityLevel.LOW,
        "desc": "QUIET_SIDEWAYS (安静震荡)"
    }
    
    # 波动率最大的状态 -> 高波动趋势
    state_labels[sorted_indices[-1]] = {
        "regime": MarketRegime.BULL if means[sorted_indices[-1], 0] > 0 else MarketRegime.BEAR, # 简单通过收益率均值正负判断方向
        "volatility": VolatilityLevel.HIGH,
        "desc": "VOLATILE_TREND (高波动趋势)"
    }
    
    # 中间状态 -> 均值回归/反转区
    if len(sorted_indices) > 2:
        state_labels[sorted_indices[1]] = {
            "regime": MarketRegime.SIDEWAYS,
            "volatility": VolatilityLevel.MEDIUM,
            "desc": "REVERSAL_ZONE (反转/过渡区)"
        }
        
    result = state_labels.get(current_state, {
        "regime": MarketRegime.SIDEWAYS,
        "volatility": VolatilityLevel.MEDIUM,
        "desc": "UNKNOWN"
    })
    
    return result["regime"], result["volatility"], result["desc"]

# @mcp.tool()
async def get_market_regime_hmm(symbol: str, timeframe: str = '1h') -> str:
    """
    使用 HMM (隐马尔可夫模型) 识别当前市场状态 (Regime)。
    """
    start_time = time.time()
    logger.info(f"Received HMM regime analysis request for {symbol} on {timeframe}")
    
    config = get_exchange_config()
    exchange = ccxt.binanceusdm(config)
    if os.getenv("USE_TESTNET", "False").lower() in ("true", "1", "yes"):
        exchange.set_sandbox_mode(True)
    
    try:
        # 1. 获取足够的历史数据用于训练 (500根)
        ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=500)
        
        if not ohlcv or len(ohlcv) < 100:
            raise ValueError(f"Insufficient OHLCV data for HMM training: {len(ohlcv) if ohlcv else 0}")
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 2. 运行 HMM 引擎
        engine = HMMEngine(n_components=3)
        
        train_start = time.time()
        hmm_result = engine.train_and_predict(df)
        train_elapsed = time.time() - train_start
        
        if train_elapsed > 0.5:
            logger.warn(f"HMM training took {train_elapsed:.4f}s (> 0.5s) for {symbol}")
            
        if not hmm_result:
            raise ValueError("HMM model failed to converge or predict.")
            
        # 3. 解读状态
        regime, volatility, state_desc = interpret_hmm_state(
            hmm_result['means'], 
            hmm_result['covars'], 
            hmm_result['current_state']
        )
        
        # 4. 生成 Insight
        insight = f"HMM 模型识别当前市场处于: {state_desc}。"
        
        if state_desc.startswith("VOLATILE_TREND"):
            insight += " 波动率正在扩张，建议缩减杠杆并切换至突破策略，注意设置移动止损。"
        elif state_desc.startswith("QUIET_SIDEWAYS"):
            insight += " 市场波动率低，适合网格交易或区间刷单策略。"
        elif state_desc.startswith("REVERSAL_ZONE"):
            insight += " 市场处于过渡阶段，可能面临变盘或均值回归，建议降低仓位观望。"
            
        # 构造 Context
        # 注意：HMM 不直接提供支撑阻力位，这里留空或复用简单计算
        atr = (df['high'] - df['low']).mean()
        
        context = MarketContext(
            symbol=symbol,
            regime=regime,
            volatility=volatility,
            atr=float(atr),
            support_levels=[], # HMM 不负责 SR
            resistance_levels=[],
            insight=f"[HMM分析] {insight}"
        )
        
        elapsed = time.time() - start_time
        logger.info(f"HMM Analysis completed for {symbol} in {elapsed:.4f}s (State: {hmm_result['current_state']})")
        
        return context.model_dump_json()

    except Exception as e:
        logger.error(f"Error in get_market_regime_hmm: {str(e)}")
        return MCPErrorResponse(
            status="error",
            error_code="HMM_ANALYSIS_ERROR",
            message=str(e)
        ).model_dump_json()
        
    finally:
        await exchange.close()

# @mcp.tool()
async def analyze_orderflow(symbol: str) -> str:
    """
    分析订单流 (Orderflow) 数据，包括 OFI 和成交主动性。
    """
    start_time = time.time()
    logger.info(f"Received Orderflow analysis request for {symbol}")
    
    config = get_exchange_config()
    exchange = ccxt.binanceusdm(config)
    if os.getenv("USE_TESTNET", "False").lower() in ("true", "1", "yes"):
        exchange.set_sandbox_mode(True)
    
    try:
        # 并发获取 盘口(Orderbook) 和 成交(Trades)
        # fetch_order_book limit=20, fetch_trades limit=100
        task_ob = exchange.fetch_order_book(symbol, limit=20)
        task_trades = exchange.fetch_trades(symbol, limit=100)
        
        orderbook, trades = await asyncio.gather(task_ob, task_trades)
        
        # 1. 计算 OFI
        ofi = orderflow_engine.calculate_ofi(symbol, orderbook)
        
        # 归一化 OFI (简单缩放，实际可能需要历史 Rolling Window Z-score)
        # 这里为了直观，假设单次 tick 变化量通常在一定范围内，简单 clip 到 -1 ~ 1 用于展示
        # 注意：这里的 OFI 是绝对量，归一化通常需要历史数据，这里仅做符号和量级参考
        # 暂且用 sigmoid 或简单分段来映射 score
        ofi_score = np.tanh(ofi / 100.0) # 假设 100 是一个较大的 volume 变动单位
        
        # 2. 分析成交主动性
        dominance_info = orderflow_engine.analyze_trades_dominance(trades)
        dominance = dominance_info['dominance']
        buy_ratio = dominance_info['buy_ratio']
        
        # 3. 综合信号
        pressure_signal = orderflow_engine.get_pressure_signal(ofi, dominance)
        
        # 4. 生成 Insight
        insight = f"Orderflow 分析: OFI={ofi:.2f}, 主动买入占比={buy_ratio:.0%} ({dominance})。"
        insight += f" 结论: {pressure_signal}。"
        
        if ofi > 0 and dominance == "BULLISH":
            insight += " [强劲看多] 挂单支撑增加且主动买盘强劲。"
        elif ofi < 0 and dominance == "BEARISH":
            insight += " [强劲看空] 挂单抛压增加且主动卖盘强劲。"
            
        # 构造返回结果 (使用 dict 转 json，或者定义新的 Pydantic 模型)
        # 这里为了灵活，直接返回结构化字典
        result = {
            "symbol": symbol,
            "ofi_raw": ofi,
            "ofi_score": float(ofi_score),
            "trade_dominance": dominance,
            "buy_ratio": buy_ratio,
            "pressure_signal": pressure_signal,
            "insight": insight,
            "timestamp": int(time.time() * 1000)
        }
        
        elapsed = time.time() - start_time
        logger.info(f"Orderflow Analysis completed for {symbol} in {elapsed:.4f}s")
        
        return json.dumps(result, ensure_ascii=False)

    except Exception as e:
        logger.error(f"Error in analyze_orderflow: {str(e)}")
        return MCPErrorResponse(
            status="error",
            error_code="ORDERFLOW_ERROR",
            message=str(e)
        ).model_dump_json()
        
    finally:
        await exchange.close()

# 1. 将 CCXT 调用抽离为被缓存的独立异步函数
@cached(cache=ohlcv_cache, key=cache_key_builder)
async def get_cached_ohlcv(ex, symbol, timeframe, limit=500):
    logger.info(f"🌐 [Cache Miss] 从交易所拉取真实数据: OHLCV {symbol}")
    return await ex.fetch_ohlcv(symbol, timeframe, limit=limit)

@cached(cache=market_cache, key=cache_key_builder)
async def get_cached_orderbook(ex, symbol, limit=20):
    return await ex.fetch_order_book(symbol, limit=limit)

@cached(cache=market_cache, key=cache_key_builder)
async def get_cached_ticker(ex, symbol):
    return await ex.fetch_ticker(symbol)

@cached(cache=market_cache, key=cache_key_builder)
async def get_cached_trades(ex, symbol, limit=200):
    return await ex.fetch_trades(symbol, limit=limit)

@mcp.tool()
async def get_full_market_context(symbol: str, timeframe: str = "1h") -> str:
    """
    [最高优先级分析工具] 一键获取指定资产的所有市场维度的脱水研报！
    当你需要分析行情时，必须且只需调用此工具一次。它会同时返回：
    当前价格、HMM市场状态、SMC结构与FVG缺口、以及Orderflow订单流不平衡(OFI)数据。
    """
    logger.info(f"🚀 启动 Fat-Tool 聚合分析: {symbol} @ {timeframe}")
    
    try:
        # 1. 核心提速点：直接从 WebSocket 内存池极速读取！
        logger.info("⚡ 正在从 WebSocket 内存池极速读取数据 (0 I/O 延迟)...")
        data = await ws_pool.get_market_data(symbol, timeframe)
        
        ohlcv = data.get('ohlcv', [])
        orderbook = data.get('orderbook', {'bids': [], 'asks': []})
        ticker = data.get('ticker', {'last': 0.0})
        trades = data.get('trades', [])
        
        # 增加防御性校验：如果蓄水失败，直接终止并通知 Agent
        if not ohlcv or not orderbook.get('bids'):
            logger.error(f"{symbol} 核心数据流缺失，无法进行矩阵运算。")
            return json.dumps({
                "error": "WebSocket 数据流尚未准备就绪",
                "system_instruction": "🚨 底层数据流连接不稳定，无法获取完整行情。请停止尝试其他参数，立刻输出 <ACTION>WAIT</ACTION> 并结束本次分析。"
            }, ensure_ascii=False)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 3. 实例化并调用算力引擎 (CPU 密集型操作)
        logger.info("🧠 正在进行高维特征矩阵运算 (HMM + SMC + OFI)...")
        
        # SMC
        smc_engine = SMCEngine(df)
        smc_res = smc_engine.analyze()
        
        # HMM
        hmm_res = hmm_engine.train_and_predict(df)
        # hmm_res 已经是 "QUIET_SIDEWAYS" 这样的字符串了
        
        # OFI & Aggressor
        ofi_score = orderflow_engine.calculate_ofi(symbol, orderbook)
        aggressor_data = orderflow_engine.analyze_aggressor_trades(trades) # 计算真实成交
        
        # ==========================================
        # 3. 记忆流对比与动态 Prompt 生成
        # ==========================================
        state_key = f"{symbol}_{timeframe}"
        previous_state = STATE_MEMORY.get(state_key)
        
        # 判断是否发生状态切换 (第一次运行也视为 true)
        is_state_changed = (previous_state != hmm_res)
        
        # 更新记忆到最新状态
        STATE_MEMORY[state_key] = hmm_res
        
        # 为大模型生成动态指令
        if is_state_changed:
            instruction = f"🚨 警告：环境已从 {previous_state or '未知'} 切换为 {hmm_res}！请立即执行【路径 B：宏观环境切换】进行深度共振重估。"
            logger.warn(f"变盘预警! {symbol} 状态从 {previous_state} -> {hmm_res}")
        else:
            instruction = f"✅ 环境稳定维持在 {hmm_res}。请执行【路径 A：环境稳定期】仅做轻量级风控校验，快速响应，禁止长篇大论。"
            logger.info(f"环境稳定 ({hmm_res})，已下发轻量级风控指令。")

        # 5. 组装终极脱水研报 (Fat Payload)
        fat_payload = {
            "asset": symbol,
            "current_price": ticker['last'],
            "state_changed": is_state_changed,
            "market_regime": hmm_res,
            "previous_regime": previous_state,
            "smart_money_concepts": smc_res,
            "orderflow": {
                "static_ofi_score": ofi_score,
                "dynamic_aggressor": aggressor_data # 注入真实的火拼数据
            },
            "system_instruction": instruction
        }
        
        logger.info("✅ Fat-Tool 聚合分析完成！脱水数据已打包。")
        return json.dumps(fat_payload, ensure_ascii=False)
        
    except ccxt.NetworkError as ne:
        logger.error(f"网络异常 (NetworkError): {ne}")
        return json.dumps({
            "error": "Network Connection Failed",
            "system_instruction": "🚨 交易所 API 连接超时或被拒绝。禁止任何分析，强制输出：<ACTION>WAIT</ACTION>"
        }, ensure_ascii=False)
    except ccxt.ExchangeError as ee:
        logger.error(f"交易所异常 (ExchangeError): {ee}")
        return json.dumps({
            "error": "Exchange API Error",
            "system_instruction": "🚨 交易所内部错误或处于维护状态。禁止任何分析，强制输出：<ACTION>WAIT</ACTION>"
        }, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Fat-Tool 聚合执行失败: {str(e)}")
        return json.dumps({"error": f"聚合分析失败: {str(e)}"}, ensure_ascii=False)

if __name__ == "__main__":
    mcp.run()
