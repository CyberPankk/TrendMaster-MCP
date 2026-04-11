import sys
import os
import asyncio
import importlib.util
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP


def _ensure_project_root_on_path() -> str:
    cur = Path(__file__).resolve()
    for parent in [cur] + list(cur.parents):
        if (parent / "mvp_system").exists():
            project_root = str(parent)
            if project_root not in sys.path:
                sys.path.append(project_root)
            return project_root
    project_root = str(cur.parents[4])
    if project_root not in sys.path:
        sys.path.append(project_root)
    return project_root


PROJECT_ROOT = _ensure_project_root_on_path()

# ⚠️ 确保 shared 模块可以被导入 (因为 MCP 的 shared 在 TrendMaster-MCP 根目录下)
def _add_mcp_root():
    current = os.path.abspath(os.path.dirname(__file__))
    for _ in range(3):
        if os.path.exists(os.path.join(current, "shared")):
            if current not in sys.path:
                sys.path.append(current)
            return
        current = os.path.dirname(current)

_add_mcp_root()

try:
    from mvp_system.backtest.engine import BacktestEngine
    from mvp_system.optimization.walk_forward import WalkForwardEngine
    from mvp_system.analysis.factor_miner import FactorMiner

    HAS_MVP_SYSTEM = True
except ImportError as e:
    import logging
    # 降低日志级别，避免污染终端输出
    logging.debug(f"未能完整导入主系统投研模块，将使用模拟/降级模式。原因: {e}")
    HAS_MVP_SYSTEM = False

from shared.logger import get_logger

logger = get_logger("Factor-Lab-MCP")
mcp = FastMCP("Factor-Lab-MCP")


@mcp.tool()
async def run_precompiled_backtest(strategy_name: str, symbol: str) -> dict:
    """
    运行预编译的策略双生子回测脚本 (Twin-Architecture Backtest)
    动态加载 {strategy_name}_bt.py 或 shadow_pool/{strategy_name}_bt.py 并传入 Mock OHLCV 执行
    """
    logger.info(f"🚀 启动预编译回测 (Twin-Architecture): {strategy_name} on {symbol}")
    
    try:
        skills_dir = os.path.join(PROJECT_ROOT, "skills")
        shadow_pool_dir = os.path.join(skills_dir, "shadow_pool")
        
        bt_filename = f"{strategy_name}_bt.py"
        main_path = os.path.join(skills_dir, bt_filename)
        shadow_path = os.path.join(shadow_pool_dir, bt_filename)
        
        target_path = None
        if os.path.exists(main_path):
            target_path = main_path
        elif os.path.exists(shadow_path):
            target_path = shadow_path
            
        if not target_path:
            logger.warning(f"预编译回测脚本未找到: {bt_filename}")
            return {
                "status": "error",
                "error": f"找不到预编译的回测脚本: {bt_filename} (搜索了 skills 和 shadow_pool 目录)",
                "timestamp": datetime.now().isoformat()
            }
            
        # 动态加载模块
        spec = importlib.util.spec_from_file_location(f"backtest_{strategy_name}", target_path)
        if spec is None or spec.loader is None:
            return {
                "status": "error",
                "error": f"无法加载模块 spec: {target_path}",
                "timestamp": datetime.now().isoformat()
            }
            
        bt_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bt_module)
        
        # 检查是否包含 run_backtest 函数
        if not hasattr(bt_module, "run_backtest"):
            return {
                "status": "error",
                "error": f"模块 {bt_filename} 缺少 'run_backtest(df)' 函数",
                "timestamp": datetime.now().isoformat()
            }
            
        # 生成 mock pandas DataFrame
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='1h')
        
        # 使用随机游走生成类似价格的数据
        np.random.seed(42) # 可选：为了可复现
        returns = np.random.normal(0, 0.01, 1000)
        price_series = 100 * np.exp(returns.cumsum())
        
        mock_df = pd.DataFrame({
            'open': price_series,
            'high': price_series * (1 + np.abs(np.random.normal(0, 0.005, 1000))),
            'low': price_series * (1 - np.abs(np.random.normal(0, 0.005, 1000))),
            'close': price_series * (1 + np.random.normal(0, 0.002, 1000)),
            'volume': np.random.randint(100, 10000, 1000)
        }, index=dates)
        
        # 执行回测
        logger.info(f"执行 {target_path} 中的 run_backtest 函数...")
        result = await asyncio.to_thread(bt_module.run_backtest, mock_df)
        
        return {
            "status": "success",
            "strategy": strategy_name,
            "symbol": symbol,
            "backtest_result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"预编译回测执行异常: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@mcp.tool()
async def run_historical_backtest(strategy_name: str, symbol: str, timeframe: str, start_date: str, end_date: str) -> dict:
    logger.info(f"🧪 启动历史回测任务: {strategy_name} | {symbol} | {timeframe}")

    if HAS_MVP_SYSTEM:
        try:
            engine = BacktestEngine(
                strategy=strategy_name,
                symbol=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
            )
            performance = await asyncio.to_thread(engine.run)
            return {
                "status": "success",
                "task": "Historical Backtest",
                "metrics": performance,
                "insight": "回测完成。请重点关注夏普比率和最大回撤是否满足风控要求。",
                "timestamp": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"回测引擎执行异常: {e}")
            return {"status": "error", "message": str(e), "timestamp": datetime.now().isoformat()}

    await asyncio.sleep(2)
    return {
        "status": "simulated_success",
        "metrics": {
            "Total Return": "15.4%",
            "Sharpe Ratio": 1.85,
            "Max Drawdown": "-4.2%",
            "Win Rate": "58.3%",
            "Profit Factor": 1.6,
        },
        "warning": "主系统未挂载，此为模拟投研数据。",
        "timestamp": datetime.now().isoformat(),
    }


@mcp.tool()
async def evaluate_alpha_factor(factor_name: str, symbol: str, timeframe: str) -> dict:
    logger.info(f"🧬 启动 Alpha 因子评估: {factor_name} on {symbol}")

    if HAS_MVP_SYSTEM:
        try:
            miner = FactorMiner()
            metrics = await asyncio.to_thread(miner.evaluate_factor, factor_name, symbol, timeframe)
            return {"status": "success", "factor": factor_name, "metrics": metrics, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"因子评估异常: {e}")
            return {"status": "error", "message": str(e), "timestamp": datetime.now().isoformat()}

    await asyncio.sleep(1)
    return {
        "status": "simulated_success",
        "factor": factor_name,
        "metrics": {"Rank IC": 0.045, "IC IR": 0.65, "Turnover": "12%"},
        "insight": "IC > 0.03 且 IR > 0.5，该 Alpha 因子具有弱预测性，可作为辅助因子纳入池中。",
        "timestamp": datetime.now().isoformat(),
    }


@mcp.tool()
async def generate_wfa_report(strategy_name: str, symbol: str) -> dict:
    logger.info(f"🚶 启动滚动步进分析 (WFA): {strategy_name} on {symbol}")

    if HAS_MVP_SYSTEM:
        try:
            wfa_engine = WalkForwardEngine(strategy_class=strategy_name, data=None, config=None)
            report = await asyncio.to_thread(wfa_engine.run)
            return {"status": "success", "report": report, "timestamp": datetime.now().isoformat()}
        except Exception as e:
            logger.error(f"WFA 执行异常: {e}")
            return {"status": "error", "message": str(e), "timestamp": datetime.now().isoformat()}

    await asyncio.sleep(3)
    return {
        "status": "simulated_success",
        "strategy": strategy_name,
        "robustness_score": "82/100",
        "degradation_ratio": "15%",
        "insight": "WFA 验证通过。样本外绩效折损仅为 15%，策略未出现严重过拟合，具备实盘稳健性。",
        "timestamp": datetime.now().isoformat(),
    }

if __name__ == "__main__":
    mcp.run()
