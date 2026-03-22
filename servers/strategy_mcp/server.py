import sys
import os
import asyncio
from typing import Dict, Any
from mcp.server.fastmcp import FastMCP

# 动态路径注入：允许子模块中的 MCP 直接导入主系统 (TrendMasterMVP) 的模块
def _add_project_root():
    """动态寻找包含 mvp_system 的根目录并加入 sys.path"""
    current = os.path.abspath(os.path.dirname(__file__))
    for _ in range(5):
        if os.path.exists(os.path.join(current, "mvp_system")):
            if current not in sys.path:
                sys.path.append(current)
            return
        current = os.path.dirname(current)

_add_project_root()

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

from shared.logger import get_logger

# 动态路径注入：指向 TrendMasterMVP 根目录
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from mvp_system.core.settings_manager import SettingsManager
    from mvp_system.core.dynamic_allocator import MetaEnsembleAllocator

    HAS_MVP_SYSTEM = True
except Exception as e:
    import logging

    logging.warning(f"未能完整导入主系统策略管理模块，将使用降级/模拟模式。原因: {e}")
    HAS_MVP_SYSTEM = False

logger = get_logger("Strategy-MCP")
mcp = FastMCP("Strategy-MCP")

MOCK_STRATEGY_POOL = {
    "Momentum_v2": {"status": "running", "allocation_pct": 0.6, "current_pnl": 0.12},
    "Market_Making_v1": {"status": "paused", "allocation_pct": 0.0, "current_pnl": -0.01},
    "Mean_Reversion": {"status": "running", "allocation_pct": 0.4, "current_pnl": 0.05},
}


@mcp.tool()
async def get_strategy_pool_status() -> dict:
    logger.info("📊 获取全局策略池与资金分配状态...")

    if HAS_MVP_SYSTEM:
        try:
            settings = SettingsManager()
            yaml_active = settings.get("strategy_manager.active_strategies", default=[], cast=list)
            yaml_shadow = settings.get("strategy_manager.shadow_strategies", default=[], cast=list)

            allocator = MetaEnsembleAllocator()

            return {
                "status": "success",
                "total_capital_usd": settings.get("initial_capital", default=0.0, cast=float),
                "active_strategies": yaml_active,
                "shadow_strategies": yaml_shadow,
                "allocator": {
                    "type": "MetaEnsembleAllocator",
                    "window": getattr(allocator, "window", None),
                    "temperature": getattr(allocator, "temperature", None),
                },
            }
        except Exception as e:
            logger.error(f"获取策略池状态失败: {e}")
            return {"status": "error", "message": str(e)}

    await asyncio.sleep(0.5)
    return {
        "status": "simulated_success",
        "total_capital_usd": 100000.0,
        "strategies": MOCK_STRATEGY_POOL,
        "insight": "趋势策略目前占用 60% 资金，但如果市场波动率下降，建议将资金向震荡策略转移。",
    }


@mcp.tool()
async def switch_active_strategy(strategy_name: str, action: str) -> dict:
    logger.info(f"🔄 策略路由切换: {action.upper()} strategy [{strategy_name}]")

    if action.lower() not in ["start", "stop", "pause"]:
        return {"error": "action 参数必须为 start, stop 或 pause"}

    if HAS_MVP_SYSTEM:
        try:
            settings = SettingsManager()

            key = "strategy_manager.active_strategies"
            current = settings.get(key, default=[], cast=list)

            if not isinstance(current, list):
                current = []

            if action.lower() == "start":
                if strategy_name not in current:
                    current.append(strategy_name)
            else:
                current = [s for s in current if s != strategy_name]

            ok = settings.set(key, current, val_type="json", user="Strategy-MCP")
            return {"status": "success" if ok else "error", "strategy": strategy_name, "action": action, "active_strategies": current}
        except Exception as e:
            logger.error(f"策略切换执行异常: {e}")
            return {"status": "error", "message": str(e)}

    await asyncio.sleep(1)
    if strategy_name in MOCK_STRATEGY_POOL:
        MOCK_STRATEGY_POOL[strategy_name]["status"] = "running" if action == "start" else "paused"
        return {"status": "simulated_success", "message": f"策略 {strategy_name} 已成功变更为 {action} 状态。"}
    return {"status": "error", "message": f"未找到策略: {strategy_name}"}


@mcp.tool()
async def rebalance_capital_allocation(strategy_weights: dict) -> dict:
    logger.info(f"⚖️ 触发资金重分配 (Rebalance): {strategy_weights}")

    total_weight = sum(strategy_weights.values())
    if not (0.99 <= total_weight <= 1.01):
        return {"error": f"资金分配比例总和必须为 1.0，当前为 {total_weight}"}

    if HAS_MVP_SYSTEM:
        try:
            settings = SettingsManager()
            ok = settings.set("strategy_manager.strategy_weights", strategy_weights, val_type="json", user="Strategy-MCP")
            return {"status": "success" if ok else "error", "message": "资金划拨参数已写入动态配置", "strategy_weights": strategy_weights}
        except Exception as e:
            logger.error(f"资金调拨异常: {e}")
            return {"status": "error", "message": str(e)}

    await asyncio.sleep(2)
    for strat, weight in strategy_weights.items():
        if strat in MOCK_STRATEGY_POOL:
            MOCK_STRATEGY_POOL[strat]["allocation_pct"] = weight
    return {
        "status": "simulated_success",
        "message": "资金划拨已生效。",
        "current_allocations": {s: v["allocation_pct"] for s, v in MOCK_STRATEGY_POOL.items()},
    }

if __name__ == "__main__":
    mcp.run()
