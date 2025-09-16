from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class TimeFrame(str, Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

class MarketStructure(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"

class TradeDirection(str, Enum):
    LONG = "long"
    SHORT = "short"

class ICTConcept(str, Enum):
    ORDER_BLOCK = "order_block"
    FAIR_VALUE_GAP = "fair_value_gap"
    BREAKER_BLOCK = "breaker_block"
    LIQUIDITY_POOL = "liquidity_pool"
    REJECTION_BLOCK = "rejection_block"
    MITIGATION_BLOCK = "mitigation_block"
    SUPPLY_DEMAND = "supply_demand"

# Market Data Models
class CandleData(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    timeframe: TimeFrame

class MarketStructurePoint(BaseModel):
    timestamp: datetime
    price: float
    type: str  # HH, HL, LH, LL
    significance: float = Field(ge=0, le=1)

# ICT Concept Models
class OrderBlock(BaseModel):
    id: Optional[str] = None
    timestamp: datetime
    high: float
    low: float
    direction: TradeDirection
    mitigation_level: Optional[float] = None
    is_mitigated: bool = False
    strength: float = Field(ge=0, le=1)
    symbol: str
    timeframe: TimeFrame

class FairValueGap(BaseModel):
    id: Optional[str] = None
    timestamp: datetime
    top: float
    bottom: float
    direction: TradeDirection
    is_filled: bool = False
    fill_percentage: float = Field(ge=0, le=1, default=0)
    symbol: str
    timeframe: TimeFrame

class LiquidityPool(BaseModel):
    id: Optional[str] = None
    timestamp: datetime
    price: float
    type: str  # equal_highs, equal_lows, trendline
    strength: float = Field(ge=0, le=1)
    is_swept: bool = False
    symbol: str
    timeframe: TimeFrame

# Trading Models
class TradeSetup(BaseModel):
    id: Optional[str] = None
    symbol: str
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: List[float]
    risk_reward_ratio: float
    setup_type: ICTConcept
    confidence: float = Field(ge=0, le=1)
    timestamp: datetime
    timeframe: TimeFrame

class TradeResult(BaseModel):
    id: Optional[str] = None
    setup_id: str
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    status: str  # pending, active, closed
    partial_fills: List[Dict[str, Any]] = []

# Backtesting Models
class BacktestRequest(BaseModel):
    symbol: str
    start_date: datetime
    end_date: datetime
    timeframe: TimeFrame
    initial_capital: float = 10000
    risk_per_trade: float = Field(ge=0.01, le=0.1, default=0.02)
    strategies: List[str]
    parameters: Dict[str, Any] = {}

class BacktestResult(BaseModel):
    id: Optional[str] = None
    request: BacktestRequest
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percentage: float
    max_drawdown: float
    max_drawdown_percentage: float
    sharpe_ratio: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    trades: List[TradeResult]
    equity_curve: List[Dict[str, Any]]
    monthly_returns: Dict[str, float]
    created_at: datetime

# AI Agent Models
class MarketAnalysis(BaseModel):
    symbol: str
    timeframe: TimeFrame
    timestamp: datetime
    market_structure: MarketStructure
    trend_direction: TradeDirection
    key_levels: List[float]
    order_blocks: List[OrderBlock]
    fair_value_gaps: List[FairValueGap]
    liquidity_pools: List[LiquidityPool]
    sentiment: str
    confidence: float = Field(ge=0, le=1)

class AIRecommendation(BaseModel):
    symbol: str
    analysis: MarketAnalysis
    trade_setups: List[TradeSetup]
    risk_assessment: str
    market_outlook: str
    key_events: List[str]
    confidence: float = Field(ge=0, le=1)
    timestamp: datetime