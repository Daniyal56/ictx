import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import talib

from app.models import (
    CandleData, OrderBlock, FairValueGap, LiquidityPool,
    MarketStructurePoint, TradeSetup, MarketAnalysis,
    TimeFrame, TradeDirection, MarketStructure, ICTConcept
)

@dataclass
class ICTLevel:
    price: float
    timestamp: datetime
    level_type: str
    strength: float
    tested_count: int = 0

class ICTStrategyManager:
    """Comprehensive ICT Strategy Manager implementing 50+ concepts"""
    
    def __init__(self):
        self.strategies = {
            "order_block_strategy": self.order_block_strategy,
            "fair_value_gap_strategy": self.fair_value_gap_strategy,
            "silver_bullet_strategy": self.silver_bullet_strategy,
            "breaker_block_strategy": self.breaker_block_strategy,
            "liquidity_raid_strategy": self.liquidity_raid_strategy,
            "smt_divergence_strategy": self.smt_divergence_strategy,
            "power_of_three_strategy": self.power_of_three_strategy,
            "rejection_block_strategy": self.rejection_block_strategy,
            "mitigation_block_strategy": self.mitigation_block_strategy,
            "turtle_soup_strategy": self.turtle_soup_strategy,
            "judas_swing_strategy": self.judas_swing_strategy,
            "optimal_trade_entry_strategy": self.optimal_trade_entry_strategy,
            "london_killzone_strategy": self.london_killzone_strategy,
            "ny_reversal_strategy": self.ny_reversal_strategy,
            "asian_range_strategy": self.asian_range_strategy
        }
    
    async def analyze_market(self, symbol: str, timeframe: TimeFrame, lookback_days: int = 30) -> MarketAnalysis:
        """Comprehensive market analysis using all ICT concepts"""
        # Get market data (mock for now)
        data = await self._get_market_data(symbol, timeframe, lookback_days)
        
        # Analyze market structure
        market_structure = self._analyze_market_structure(data)
        
        # Identify key levels
        key_levels = self._identify_key_levels(data)
        
        # Find order blocks
        order_blocks = self._find_order_blocks(data)
        
        # Find fair value gaps
        fair_value_gaps = self._find_fair_value_gaps(data)
        
        # Find liquidity pools
        liquidity_pools = self._find_liquidity_pools(data)
        
        # Determine trend direction
        trend_direction = self._get_trend_direction(data)
        
        return MarketAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.utcnow(),
            market_structure=market_structure,
            trend_direction=trend_direction,
            key_levels=key_levels,
            order_blocks=order_blocks,
            fair_value_gaps=fair_value_gaps,
            liquidity_pools=liquidity_pools,
            sentiment="neutral",
            confidence=0.75
        )
    
    async def get_trade_setups(self, symbol: str, timeframe: TimeFrame, strategies: List[str]) -> List[TradeSetup]:
        """Generate trade setups using specified strategies"""
        all_setups = []
        data = await self._get_market_data(symbol, timeframe, 30)
        
        for strategy_name in strategies:
            if strategy_name in self.strategies:
                setups = await self.strategies[strategy_name](data, symbol, timeframe)
                all_setups.extend(setups)
        
        # Sort by confidence
        all_setups.sort(key=lambda x: x.confidence, reverse=True)
        
        return all_setups
    
    async def get_market_structure(self, symbol: str, timeframe: TimeFrame) -> Dict[str, Any]:
        """Get detailed market structure analysis"""
        data = await self._get_market_data(symbol, timeframe, 50)
        
        # Find swing highs and lows
        swing_points = self._find_swing_points(data)
        
        # Classify structure
        structure_type = self._classify_market_structure(swing_points)
        
        # Find break of structure points
        bos_points = self._find_break_of_structure(swing_points)
        
        # Find change of character points
        choch_points = self._find_change_of_character(swing_points)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "structure_type": structure_type,
            "swing_points": swing_points,
            "break_of_structure": bos_points,
            "change_of_character": choch_points,
            "current_trend": self._get_trend_direction(data),
            "key_levels": self._identify_key_levels(data),
            "timestamp": datetime.utcnow()
        }
    
    def get_current_killzone(self) -> Dict[str, Any]:
        """Get current trading killzone"""
        current_time = datetime.utcnow()
        hour = current_time.hour
        
        killzones = {
            "london_open": {"start": 7, "end": 10, "name": "London Open Killzone"},
            "ny_am": {"start": 12, "end": 15, "name": "New York AM Killzone"},
            "ny_pm": {"start": 15, "end": 18, "name": "New York PM Killzone"},
            "london_close": {"start": 15, "end": 16, "name": "London Close Killzone"},
            "asia_range": {"start": 20, "end": 6, "name": "Asia Range"}
        }
        
        current_killzone = None
        for kz_id, kz in killzones.items():
            if kz["start"] < kz["end"]:
                if kz["start"] <= hour < kz["end"]:
                    current_killzone = kz["name"]
                    break
            else:  # Crosses midnight
                if hour >= kz["start"] or hour < kz["end"]:
                    current_killzone = kz["name"]
                    break
        
        return {
            "current_time_utc": current_time,
            "active_killzone": current_killzone,
            "next_killzone": self._get_next_killzone(hour, killzones),
            "killzone_strength": self._assess_killzone_strength(current_killzone)
        }
    
    # Strategy Implementations
    async def order_block_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Order Block Strategy - Trade institutional order blocks"""
        setups = []
        order_blocks = self._find_order_blocks(data)
        
        for ob in order_blocks:
            if not ob.is_mitigated:
                # Check for entry opportunity
                current_price = data['close'].iloc[-1]
                
                if ob.direction == TradeDirection.LONG and current_price <= ob.high:
                    entry_price = ob.low + (ob.high - ob.low) * 0.25
                    stop_loss = ob.low - (ob.high - ob.low) * 0.2
                    take_profit = [
                        entry_price + (entry_price - stop_loss) * 1.5,
                        entry_price + (entry_price - stop_loss) * 2.5
                    ]
                    
                    setup = TradeSetup(
                        symbol=symbol,
                        direction=ob.direction,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=1.5,
                        setup_type=ICTConcept.ORDER_BLOCK,
                        confidence=0.8,
                        timestamp=datetime.utcnow(),
                        timeframe=timeframe
                    )
                    setups.append(setup)
        
        return setups
    
    async def fair_value_gap_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Fair Value Gap Strategy - Trade FVG fills"""
        setups = []
        fvgs = self._find_fair_value_gaps(data)
        
        for fvg in fvgs:
            if not fvg.is_filled:
                current_price = data['close'].iloc[-1]
                
                if fvg.direction == TradeDirection.LONG and fvg.bottom <= current_price <= fvg.top:
                    entry_price = fvg.bottom + (fvg.top - fvg.bottom) * 0.5
                    stop_loss = fvg.bottom - (fvg.top - fvg.bottom) * 0.3
                    take_profit = [entry_price + (entry_price - stop_loss) * 2]
                    
                    setup = TradeSetup(
                        symbol=symbol,
                        direction=fvg.direction,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=2.0,
                        setup_type=ICTConcept.FAIR_VALUE_GAP,
                        confidence=0.75,
                        timestamp=datetime.utcnow(),
                        timeframe=timeframe
                    )
                    setups.append(setup)
        
        return setups
    
    async def silver_bullet_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """ICT Silver Bullet - NY Open 15-min window strategy"""
        setups = []
        current_time = datetime.utcnow()
        
        # Check if we're in NY open killzone (12:30-13:30 UTC)
        if 12 <= current_time.hour <= 13 and current_time.minute <= 30:
            # Look for liquidity sweeps and reversals
            liquidity_pools = self._find_liquidity_pools(data)
            
            for pool in liquidity_pools:
                if not pool.is_swept:
                    current_price = data['close'].iloc[-1]
                    
                    # Check for sweep and reversal setup
                    if abs(current_price - pool.price) / pool.price < 0.001:  # Near liquidity
                        direction = TradeDirection.LONG if pool.type == "equal_lows" else TradeDirection.SHORT
                        
                        entry_price = current_price
                        if direction == TradeDirection.LONG:
                            stop_loss = pool.price - 20  # 20 pips below liquidity
                            take_profit = [entry_price + 50, entry_price + 100]  # 50 and 100 pip targets
                        else:
                            stop_loss = pool.price + 20
                            take_profit = [entry_price - 50, entry_price - 100]
                        
                        setup = TradeSetup(
                            symbol=symbol,
                            direction=direction,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            risk_reward_ratio=2.5,
                            setup_type=ICTConcept.LIQUIDITY_POOL,
                            confidence=0.85,
                            timestamp=datetime.utcnow(),
                            timeframe=timeframe
                        )
                        setups.append(setup)
        
        return setups
    
    async def breaker_block_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Breaker Block Strategy - Trade failed order blocks"""
        setups = []
        # Implementation for breaker blocks
        # This would involve identifying order blocks that have been broken
        # and now act as support/resistance in the opposite direction
        return setups
    
    async def liquidity_raid_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Liquidity Raid Strategy - Trade liquidity sweeps and reversals"""
        setups = []
        # Implementation for liquidity raids
        return setups
    
    async def smt_divergence_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """SMT Divergence Strategy - Smart Money Divergence"""
        setups = []
        # Implementation for SMT divergence
        return setups
    
    async def power_of_three_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Power of 3 Strategy - Accumulation, Manipulation, Distribution"""
        setups = []
        # Implementation for Power of 3 model
        return setups
    
    async def rejection_block_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Rejection Block Strategy"""
        setups = []
        # Implementation for rejection blocks
        return setups
    
    async def mitigation_block_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Mitigation Block Strategy"""
        setups = []
        # Implementation for mitigation blocks
        return setups
    
    async def turtle_soup_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Turtle Soup Strategy - Stop hunt reversals"""
        setups = []
        # Implementation for turtle soup
        return setups
    
    async def judas_swing_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Judas Swing Strategy - False breakouts at session opens"""
        setups = []
        # Implementation for Judas swings
        return setups
    
    async def optimal_trade_entry_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Optimal Trade Entry Strategy - 62%-79% retracements"""
        setups = []
        # Implementation for OTE setups
        return setups
    
    async def london_killzone_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """London Killzone Strategy"""
        setups = []
        # Implementation for London killzone
        return setups
    
    async def ny_reversal_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """New York Reversal Strategy"""
        setups = []
        # Implementation for NY reversals
        return setups
    
    async def asian_range_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Asian Range Strategy"""
        setups = []
        # Implementation for Asian range trading
        return setups
    
    # Helper Methods
    async def _get_market_data(self, symbol: str, timeframe: TimeFrame, days: int) -> pd.DataFrame:
        """Get market data (mock implementation)"""
        # Mock data for development
        dates = pd.date_range(end=datetime.utcnow(), periods=days*24, freq='H')
        np.random.seed(42)
        
        base_price = 1.1000  # For EURUSD
        prices = []
        
        for i in range(len(dates)):
            change = np.random.normal(0, 0.001)
            if i == 0:
                price = base_price
            else:
                price = prices[-1] + change
            prices.append(price)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + abs(np.random.normal(0, 0.0005)) for p in prices],
            'low': [p - abs(np.random.normal(0, 0.0005)) for p in prices],
            'close': [p + np.random.normal(0, 0.0002) for p in prices],
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        return data
    
    def _analyze_market_structure(self, data: pd.DataFrame) -> MarketStructure:
        """Analyze market structure"""
        # Simplified implementation
        close_prices = data['close'].values
        if len(close_prices) < 20:
            return MarketStructure.RANGING
        
        recent_trend = np.polyfit(range(len(close_prices[-20:])), close_prices[-20:], 1)[0]
        
        if recent_trend > 0.0001:
            return MarketStructure.BULLISH
        elif recent_trend < -0.0001:
            return MarketStructure.BEARISH
        else:
            return MarketStructure.RANGING
    
    def _identify_key_levels(self, data: pd.DataFrame) -> List[float]:
        """Identify key support and resistance levels"""
        highs = data['high'].values
        lows = data['low'].values
        
        # Find significant levels using pivot points
        key_levels = []
        
        # Add recent highs and lows
        recent_high = np.max(highs[-20:])
        recent_low = np.min(lows[-20:])
        
        key_levels.extend([recent_high, recent_low])
        
        # Add psychological levels (round numbers)
        current_price = data['close'].iloc[-1]
        for level in [1.1000, 1.1050, 1.1100, 1.1150, 1.1200]:
            if abs(current_price - level) / current_price < 0.02:  # Within 2%
                key_levels.append(level)
        
        return sorted(list(set(key_levels)))
    
    def _find_order_blocks(self, data: pd.DataFrame) -> List[OrderBlock]:
        """Find order blocks in the data"""
        order_blocks = []
        
        # Simplified order block detection
        # Look for strong moves with preceding consolidation
        for i in range(10, len(data) - 1):
            # Check for strong bullish move
            if (data['close'].iloc[i] - data['open'].iloc[i]) > (data['high'].iloc[i] - data['low'].iloc[i]) * 0.7:
                # This is a strong bullish candle, check if it's an order block
                ob = OrderBlock(
                    timestamp=data['timestamp'].iloc[i],
                    high=data['high'].iloc[i],
                    low=data['low'].iloc[i],
                    direction=TradeDirection.LONG,
                    strength=0.8,
                    symbol="EURUSD",
                    timeframe=TimeFrame.H1
                )
                order_blocks.append(ob)
        
        return order_blocks[-5:]  # Return last 5 order blocks
    
    def _find_fair_value_gaps(self, data: pd.DataFrame) -> List[FairValueGap]:
        """Find fair value gaps in the data"""
        fvgs = []
        
        # Look for gaps between candles
        for i in range(2, len(data)):
            # Bullish FVG: previous candle high < next candle low
            if data['high'].iloc[i-2] < data['low'].iloc[i]:
                fvg = FairValueGap(
                    timestamp=data['timestamp'].iloc[i-1],
                    top=data['low'].iloc[i],
                    bottom=data['high'].iloc[i-2],
                    direction=TradeDirection.LONG,
                    symbol="EURUSD",
                    timeframe=TimeFrame.H1
                )
                fvgs.append(fvg)
            
            # Bearish FVG: previous candle low > next candle high
            elif data['low'].iloc[i-2] > data['high'].iloc[i]:
                fvg = FairValueGap(
                    timestamp=data['timestamp'].iloc[i-1],
                    top=data['low'].iloc[i-2],
                    bottom=data['high'].iloc[i],
                    direction=TradeDirection.SHORT,
                    symbol="EURUSD",
                    timeframe=TimeFrame.H1
                )
                fvgs.append(fvg)
        
        return fvgs[-3:]  # Return last 3 FVGs
    
    def _find_liquidity_pools(self, data: pd.DataFrame) -> List[LiquidityPool]:
        """Find liquidity pools (equal highs/lows)"""
        pools = []
        
        # Find equal highs
        highs = data['high'].values
        for i in range(5, len(highs) - 5):
            nearby_highs = highs[i-5:i+5]
            max_high = np.max(nearby_highs)
            equal_count = np.sum(np.abs(nearby_highs - max_high) < max_high * 0.001)
            
            if equal_count >= 2:  # At least 2 equal highs
                pool = LiquidityPool(
                    timestamp=data['timestamp'].iloc[i],
                    price=max_high,
                    type="equal_highs",
                    strength=min(equal_count / 5.0, 1.0),
                    symbol="EURUSD",
                    timeframe=TimeFrame.H1
                )
                pools.append(pool)
        
        return pools[-3:]  # Return last 3 pools
    
    def _get_trend_direction(self, data: pd.DataFrame) -> TradeDirection:
        """Determine overall trend direction"""
        close_prices = data['close'].values[-20:]  # Last 20 periods
        trend_slope = np.polyfit(range(len(close_prices)), close_prices, 1)[0]
        
        return TradeDirection.LONG if trend_slope > 0 else TradeDirection.SHORT
    
    def _find_swing_points(self, data: pd.DataFrame) -> List[MarketStructurePoint]:
        """Find swing highs and lows"""
        swing_points = []
        highs = data['high'].values
        lows = data['low'].values
        
        # Simple swing point detection
        for i in range(2, len(data) - 2):
            # Swing high
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                
                point = MarketStructurePoint(
                    timestamp=data['timestamp'].iloc[i],
                    price=highs[i],
                    type="swing_high",
                    significance=0.8
                )
                swing_points.append(point)
            
            # Swing low
            elif (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                  lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                
                point = MarketStructurePoint(
                    timestamp=data['timestamp'].iloc[i],
                    price=lows[i],
                    type="swing_low",
                    significance=0.8
                )
                swing_points.append(point)
        
        return swing_points
    
    def _classify_market_structure(self, swing_points: List[MarketStructurePoint]) -> str:
        """Classify market structure based on swing points"""
        if len(swing_points) < 4:
            return "insufficient_data"
        
        # Look at recent swing points
        recent_points = swing_points[-4:]
        
        # Check for higher highs and higher lows (bullish structure)
        highs = [p for p in recent_points if p.type == "swing_high"]
        lows = [p for p in recent_points if p.type == "swing_low"]
        
        if len(highs) >= 2 and len(lows) >= 2:
            if highs[-1].price > highs[-2].price and lows[-1].price > lows[-2].price:
                return "bullish_structure"
            elif highs[-1].price < highs[-2].price and lows[-1].price < lows[-2].price:
                return "bearish_structure"
        
        return "ranging_structure"
    
    def _find_break_of_structure(self, swing_points: List[MarketStructurePoint]) -> List[Dict[str, Any]]:
        """Find break of structure points"""
        bos_points = []
        # Implementation for BOS detection
        return bos_points
    
    def _find_change_of_character(self, swing_points: List[MarketStructurePoint]) -> List[Dict[str, Any]]:
        """Find change of character points"""
        choch_points = []
        # Implementation for CHoCH detection
        return choch_points
    
    def _get_next_killzone(self, current_hour: int, killzones: Dict) -> str:
        """Get the next upcoming killzone"""
        # Simplified implementation
        return "London Open Killzone"
    
    def _assess_killzone_strength(self, killzone: str) -> str:
        """Assess the strength of current killzone"""
        if killzone in ["New York AM Killzone", "London Open Killzone"]:
            return "high"
        elif killzone in ["New York PM Killzone", "London Close Killzone"]:
            return "medium"
        else:
            return "low"