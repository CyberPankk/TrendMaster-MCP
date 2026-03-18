from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from enum import Enum
from datetime import datetime

# --- 1. 基础枚举定义 ---

class MarketRegime(str, Enum):
    BULL = "BULL"          # 牛市/上升趋势
    BEAR = "BEAR"          # 熊市/下降趋势
    SIDEWAYS = "SIDEWAYS"  # 震荡/横盘

class VolatilityLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

# --- 2. 原子化数据契约 ---

class TickerData(BaseModel):
    """标准化实时行情"""
    symbol: str = Field(..., description="交易对名称，统一格式如 BTC/USDT")
    last_price: float = Field(..., description="最新成交价")
    change_24h_pct: float = Field(..., description="24小时涨跌幅百分比")
    high_24h: float = Field(..., description="24小时最高价")
    low_24h: float = Field(..., description="24小时最低价")
    volume_24h: float = Field(..., description="24小时成交额(Quote Asset)")
    timestamp: int = Field(..., description="毫秒级时间戳")

class OrderbookSnapshot(BaseModel):
    """盘口深度脱水快照"""
    symbol: str
    best_bid: float = Field(..., description="买一价")
    best_ask: float = Field(..., description="卖一价")
    ofi: float = Field(..., description="订单流不平衡指标 (Order Flow Imbalance)")
    bid_ask_spread: float = Field(..., description="买卖价差")
    depth_summary: str = Field(..., description="盘口压力的自然语言描述")

# --- 3. 高维分析契约 (给 Agent 的决策依据) ---

class MarketContext(BaseModel):
    """由 Market-MCP 生成的市场环境分析结论"""
    symbol: str
    regime: MarketRegime
    volatility: VolatilityLevel
    atr: float = Field(..., description="平均真实波幅，用于设置止损")
    support_levels: List[float] = Field(default_factory=list, description="近期关键支撑位")
    resistance_levels: List[float] = Field(default_factory=list, description="近期关键阻力位")
    insight: str = Field(..., description="给 Agent 的一段话总结，如：目前处于缩量震荡，建议观望。")

# --- 4. 衍生品特供契约 ---

class DerivativeSentiment(BaseModel):
    """合约市场情绪指标"""
    symbol: str
    funding_rate: float = Field(..., description="当前资金费率")
    oi_change_24h: float = Field(..., description="未平仓合约24H变化率")
    long_short_ratio: Optional[float] = Field(None, description="多空比")
    is_crowded: bool = Field(False, description="多头或空头是否过于拥挤")

# --- 5. 通用异常响应契约 ---

class MCPErrorResponse(BaseModel):
    """所有 MCP Tool 统一的错误反馈格式"""
    status: str = "error"
    error_code: str  # 例如: EXCHANGE_TIMEOUT, INVALID_SYMBOL
    message: str     # 给 Agent 看的错误描述
