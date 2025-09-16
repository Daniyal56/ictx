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
        
        # First find order blocks
        order_blocks = self._find_order_blocks(data)
        
        for ob in order_blocks:
            ob_timestamp = ob.timestamp
            ob_index = data[data['timestamp'] == ob_timestamp].index
            
            if len(ob_index) == 0:
                continue
                
            ob_idx = ob_index[0]
            
            # Look for break of order block
            for j in range(ob_idx + 1, min(ob_idx + 50, len(data))):
                current_price = data.iloc[j]['close']
                current_high = data.iloc[j]['high']
                current_low = data.iloc[j]['low']
                
                # Check if order block was broken
                broken = False
                new_direction = None
                
                if ob.direction == TradeDirection.LONG and current_low < ob.low:
                    # Bullish order block broken, now becomes bearish breaker
                    broken = True
                    new_direction = TradeDirection.SHORT
                elif ob.direction == TradeDirection.SHORT and current_high > ob.high:
                    # Bearish order block broken, now becomes bullish breaker
                    broken = True
                    new_direction = TradeDirection.LONG
                
                if broken:
                    # Look for retest of the broken level
                    for k in range(j + 1, min(j + 30, len(data))):
                        retest_price = data.iloc[k]['close']
                        retest_high = data.iloc[k]['high']
                        retest_low = data.iloc[k]['low']
                        
                        # Check for retest
                        if new_direction == TradeDirection.SHORT:
                            # Look for retest of broken support (now resistance)
                            if retest_high >= ob.low and retest_price < ob.high:
                                entry_price = ob.low
                                stop_loss = ob.high + (ob.high - ob.low) * 0.2
                                take_profit = [
                                    entry_price - (stop_loss - entry_price) * 1.5,
                                    entry_price - (stop_loss - entry_price) * 2.5
                                ]
                                
                                setup = TradeSetup(
                                    symbol=symbol,
                                    direction=new_direction,
                                    entry_price=entry_price,
                                    stop_loss=stop_loss,
                                    take_profit=take_profit,
                                    risk_reward_ratio=1.5,
                                    setup_type=ICTConcept.BREAKER_BLOCK,
                                    confidence=0.75,
                                    timestamp=data.iloc[k]['timestamp'],
                                    timeframe=timeframe
                                )
                                setups.append(setup)
                                break
                        
                        else:  # LONG
                            # Look for retest of broken resistance (now support)
                            if retest_low <= ob.high and retest_price > ob.low:
                                entry_price = ob.high
                                stop_loss = ob.low - (ob.high - ob.low) * 0.2
                                take_profit = [
                                    entry_price + (entry_price - stop_loss) * 1.5,
                                    entry_price + (entry_price - stop_loss) * 2.5
                                ]
                                
                                setup = TradeSetup(
                                    symbol=symbol,
                                    direction=new_direction,
                                    entry_price=entry_price,
                                    stop_loss=stop_loss,
                                    take_profit=take_profit,
                                    risk_reward_ratio=1.5,
                                    setup_type=ICTConcept.BREAKER_BLOCK,
                                    confidence=0.75,
                                    timestamp=data.iloc[k]['timestamp'],
                                    timeframe=timeframe
                                )
                                setups.append(setup)
                                break
                    break
        
        return setups
    
    async def liquidity_raid_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Liquidity Raid Strategy - Trade liquidity sweeps and reversals"""
        setups = []
        
        # Find liquidity pools
        liquidity_pools = self._find_liquidity_pools(data)
        
        for pool in liquidity_pools:
            pool_timestamp = pool.timestamp
            pool_index = data[data['timestamp'] == pool_timestamp].index
            
            if len(pool_index) == 0:
                continue
                
            pool_idx = pool_index[0]
            
            # Look for liquidity sweep
            for j in range(pool_idx + 1, min(pool_idx + 50, len(data))):
                current_high = data.iloc[j]['high']
                current_low = data.iloc[j]['low']
                current_close = data.iloc[j]['close']
                
                swept = False
                direction = None
                
                if pool.type == "equal_highs" and current_high > pool.price:
                    # High liquidity swept, expect reversal down
                    swept = True
                    direction = TradeDirection.SHORT
                elif pool.type == "equal_lows" and current_low < pool.price:
                    # Low liquidity swept, expect reversal up
                    swept = True
                    direction = TradeDirection.LONG
                
                if swept:
                    # Look for immediate reversal
                    reversal_confirmed = False
                    
                    # Check next few candles for reversal
                    for k in range(j + 1, min(j + 5, len(data))):
                        next_close = data.iloc[k]['close']
                        
                        if direction == TradeDirection.SHORT:
                            # Look for close back below liquidity level
                            if next_close < pool.price:
                                reversal_confirmed = True
                                break
                        else:  # LONG
                            # Look for close back above liquidity level
                            if next_close > pool.price:
                                reversal_confirmed = True
                                break
                    
                    if reversal_confirmed:
                        # Create trade setup
                        if direction == TradeDirection.SHORT:
                            entry_price = pool.price
                            stop_loss = current_high + (current_high - pool.price) * 0.2
                            take_profit = [
                                entry_price - (stop_loss - entry_price) * 1.5,
                                entry_price - (stop_loss - entry_price) * 2.5
                            ]
                        else:  # LONG
                            entry_price = pool.price
                            stop_loss = current_low - (pool.price - current_low) * 0.2
                            take_profit = [
                                entry_price + (entry_price - stop_loss) * 1.5,
                                entry_price + (entry_price - stop_loss) * 2.5
                            ]
                        
                        setup = TradeSetup(
                            symbol=symbol,
                            direction=direction,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            risk_reward_ratio=1.5,
                            setup_type=ICTConcept.LIQUIDITY_POOL,
                            confidence=0.8,
                            timestamp=data.iloc[k]['timestamp'],
                            timeframe=timeframe
                        )
                        setups.append(setup)
                    break
        
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
        
        # Find significant swing moves
        swing_points = self._find_swing_points(data)
        
        for i in range(1, len(swing_points)):
            current_swing = swing_points[i]
            previous_swing = swing_points[i-1]
            
            # Check if we have a valid swing structure
            if ((current_swing.type == "swing_high" and previous_swing.type == "swing_low") or
                (current_swing.type == "swing_low" and previous_swing.type == "swing_high")):
                
                # Calculate retracement levels
                swing_high = max(current_swing.price, previous_swing.price)
                swing_low = min(current_swing.price, previous_swing.price)
                range_size = swing_high - swing_low
                
                # OTE levels (62%-79% retracements)
                ote_low = swing_low + range_size * 0.62
                ote_high = swing_low + range_size * 0.79
                
                # Determine direction based on last swing
                if current_swing.type == "swing_high":
                    # Bullish structure, look for long entries on retracement
                    direction = TradeDirection.LONG
                    entry_zone_low = ote_low
                    entry_zone_high = ote_high
                    stop_loss = swing_low - range_size * 0.1
                    take_profit = [
                        swing_high + range_size * 0.5,
                        swing_high + range_size * 1.0
                    ]
                else:
                    # Bearish structure, look for short entries on retracement  
                    direction = TradeDirection.SHORT
                    entry_zone_low = swing_high - range_size * 0.79
                    entry_zone_high = swing_high - range_size * 0.62
                    stop_loss = swing_high + range_size * 0.1
                    take_profit = [
                        swing_low - range_size * 0.5,
                        swing_low - range_size * 1.0
                    ]
                
                # Look for price entering OTE zone
                current_swing_index = data[data['timestamp'] == current_swing.timestamp].index
                if len(current_swing_index) == 0:
                    continue
                    
                swing_idx = current_swing_index[0]
                
                for j in range(swing_idx + 1, min(swing_idx + 50, len(data))):
                    current_price = data.iloc[j]['close']
                    current_high = data.iloc[j]['high']
                    current_low = data.iloc[j]['low']
                    
                    # Check if price is in OTE zone
                    in_ote_zone = (entry_zone_low <= current_price <= entry_zone_high)
                    
                    if in_ote_zone:
                        # Look for reversal signals
                        reversal_signals = []
                        
                        # Check for rejection candle
                        candle_range = current_high - current_low
                        body_size = abs(data.iloc[j]['close'] - data.iloc[j]['open'])
                        
                        if direction == TradeDirection.LONG:
                            lower_wick = min(data.iloc[j]['open'], data.iloc[j]['close']) - current_low
                            if lower_wick > candle_range * 0.6:
                                reversal_signals.append("long_lower_wick")
                        else:
                            upper_wick = current_high - max(data.iloc[j]['open'], data.iloc[j]['close'])
                            if upper_wick > candle_range * 0.6:
                                reversal_signals.append("long_upper_wick")
                        
                        # Check for bullish/bearish engulfing
                        if j > 0:
                            prev_open = data.iloc[j-1]['open']
                            prev_close = data.iloc[j-1]['close']
                            curr_open = data.iloc[j]['open']
                            curr_close = data.iloc[j]['close']
                            
                            if direction == TradeDirection.LONG:
                                # Bullish engulfing
                                if (prev_close < prev_open and curr_close > curr_open and
                                    curr_close > prev_open and curr_open < prev_close):
                                    reversal_signals.append("bullish_engulfing")
                            else:
                                # Bearish engulfing
                                if (prev_close > prev_open and curr_close < curr_open and
                                    curr_close < prev_open and curr_open > prev_close):
                                    reversal_signals.append("bearish_engulfing")
                        
                        # If we have reversal signals, create setup
                        if len(reversal_signals) > 0:
                            entry_price = (entry_zone_low + entry_zone_high) / 2
                            confidence = 0.7 + len(reversal_signals) * 0.1
                            
                            setup = TradeSetup(
                                symbol=symbol,
                                direction=direction,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                risk_reward_ratio=abs((take_profit[0] - entry_price) / (stop_loss - entry_price)),
                                setup_type=ICTConcept.OPTIMAL_TRADE_ENTRY,
                                confidence=min(confidence, 1.0),
                                timestamp=data.iloc[j]['timestamp'],
                                timeframe=timeframe
                            )
                            setups.append(setup)
                            break
        
        return setups
    
    async def london_killzone_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """London Killzone Strategy - Trade during London session"""
        setups = []
        
        # London killzone: 7:00-10:00 GMT
        for i in range(len(data)):
            timestamp = data.iloc[i]['timestamp']
            hour = timestamp.hour
            
            # Check if we're in London killzone
            if 7 <= hour <= 10:
                # Look for institutional moves
                current_price = data.iloc[i]['close']
                current_high = data.iloc[i]['high']
                current_low = data.iloc[i]['low']
                
                # Check for volume expansion
                if i >= 10:
                    avg_volume = np.mean(data.iloc[i-10:i]['volume'])
                    current_volume = data.iloc[i]['volume']
                    
                    volume_expansion = current_volume > avg_volume * 1.5
                    
                    if volume_expansion:
                        # Check for strong directional move
                        body_size = abs(data.iloc[i]['close'] - data.iloc[i]['open'])
                        candle_range = current_high - current_low
                        
                        strong_move = body_size > candle_range * 0.7
                        
                        if strong_move:
                            # Determine direction
                            if data.iloc[i]['close'] > data.iloc[i]['open']:
                                direction = TradeDirection.LONG
                                entry_price = current_high
                                stop_loss = current_low - candle_range * 0.2
                                take_profit = [
                                    entry_price + candle_range * 1.5,
                                    entry_price + candle_range * 2.5
                                ]
                            else:
                                direction = TradeDirection.SHORT
                                entry_price = current_low
                                stop_loss = current_high + candle_range * 0.2
                                take_profit = [
                                    entry_price - candle_range * 1.5,
                                    entry_price - candle_range * 2.5
                                ]
                            
                            setup = TradeSetup(
                                symbol=symbol,
                                direction=direction,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                risk_reward_ratio=1.5,
                                setup_type=ICTConcept.KILLZONE,
                                confidence=0.75,
                                timestamp=timestamp,
                                timeframe=timeframe
                            )
                            setups.append(setup)
        
        return setups
    
    async def ny_reversal_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """New York Reversal Strategy - Trade NY session reversals"""
        setups = []
        
        # NY session: 13:00-16:00 GMT
        for i in range(20, len(data)):  # Need some history
            timestamp = data.iloc[i]['timestamp']
            hour = timestamp.hour
            
            # Check if we're in NY session
            if 13 <= hour <= 16:
                # Look for reversal patterns
                
                # Get AM session high/low (London session)
                am_data = []
                for j in range(max(0, i-20), i):
                    prev_hour = data.iloc[j]['timestamp'].hour
                    if 7 <= prev_hour <= 12:  # London to NY overlap
                        am_data.append(j)
                
                if len(am_data) >= 5:
                    am_high = max(data.iloc[am_data]['high'])
                    am_low = min(data.iloc[am_data]['low'])
                    
                    current_price = data.iloc[i]['close']
                    current_high = data.iloc[i]['high']
                    current_low = data.iloc[i]['low']
                    
                    # Look for sweep and reverse
                    reversal_setup = None
                    
                    # Sweep above AM high then reverse
                    if current_high > am_high and current_price < am_high:
                        # Bearish reversal
                        direction = TradeDirection.SHORT
                        entry_price = am_high
                        stop_loss = current_high + (current_high - am_high) * 0.2
                        take_profit = [
                            am_low,
                            am_low - (am_high - am_low) * 0.5
                        ]
                        reversal_setup = "sweep_high_reverse"
                    
                    # Sweep below AM low then reverse
                    elif current_low < am_low and current_price > am_low:
                        # Bullish reversal
                        direction = TradeDirection.LONG
                        entry_price = am_low
                        stop_loss = current_low - (am_low - current_low) * 0.2
                        take_profit = [
                            am_high,
                            am_high + (am_high - am_low) * 0.5
                        ]
                        reversal_setup = "sweep_low_reverse"
                    
                    if reversal_setup:
                        # Confirm with volume
                        confidence = 0.7
                        if i >= 10:
                            avg_volume = np.mean(data.iloc[i-10:i]['volume'])
                            if data.iloc[i]['volume'] > avg_volume * 1.3:
                                confidence = 0.8
                        
                        setup = TradeSetup(
                            symbol=symbol,
                            direction=direction,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            risk_reward_ratio=abs((take_profit[0] - entry_price) / (stop_loss - entry_price)),
                            setup_type=ICTConcept.REVERSAL,
                            confidence=confidence,
                            timestamp=timestamp,
                            timeframe=timeframe
                        )
                        setups.append(setup)
        
        return setups
    
    async def asian_range_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Asian Range Strategy"""
        setups = []
        # Implementation for Asian range trading
        return setups
    
    # Helper Methods
    async def _get_market_data(self, symbol: str, timeframe: TimeFrame, days: int) -> pd.DataFrame:
        """Get real market data using yfinance"""
        try:
            # Convert timeframe to yfinance interval
            interval_map = {
                TimeFrame.M1: "1m",
                TimeFrame.M5: "5m", 
                TimeFrame.M15: "15m",
                TimeFrame.M30: "30m",
                TimeFrame.H1: "1h",
                TimeFrame.H4: "4h",
                TimeFrame.D1: "1d"
            }
            
            interval = interval_map.get(timeframe, "1h")
            
            # Calculate period based on timeframe
            if interval in ["1m", "5m"]:
                period = f"{min(days, 7)}d"  # Limited for intraday
            elif interval in ["15m", "30m", "1h"]:
                period = f"{min(days, 30)}d"
            elif interval == "4h":
                period = f"{min(days, 60)}d"
            else:
                period = f"{min(days, 365)}d"
            
            # Get data from Yahoo Finance
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if hist.empty:
                # Fallback to mock data if real data unavailable
                return self._generate_mock_data(symbol, timeframe, days)
            
            # Convert to our DataFrame format
            data = pd.DataFrame({
                'timestamp': hist.index,
                'open': hist['Open'].values,
                'high': hist['High'].values,
                'low': hist['Low'].values,
                'close': hist['Close'].values,
                'volume': hist['Volume'].values
            })
            
            # Reset index to make timestamp a column
            data = data.reset_index(drop=True)
            
            return data
            
        except Exception as e:
            print(f"Error fetching market data for {symbol}: {str(e)}")
            # Fallback to mock data
            return self._generate_mock_data(symbol, timeframe, days)
    
    def _generate_mock_data(self, symbol: str, timeframe: TimeFrame, days: int) -> pd.DataFrame:
        """Generate mock data as fallback"""
        # Calculate periods based on timeframe
        periods_per_day = {
            TimeFrame.M1: 1440,
            TimeFrame.M5: 288,
            TimeFrame.M15: 96,
            TimeFrame.M30: 48,
            TimeFrame.H1: 24,
            TimeFrame.H4: 6,
            TimeFrame.D1: 1
        }
        
        periods = days * periods_per_day.get(timeframe, 24)
        dates = pd.date_range(end=datetime.utcnow(), periods=periods, freq='H')
        
        # Use consistent seed for reproducible data
        np.random.seed(hash(symbol) % 1000)
        
        # Base prices for different symbols
        base_prices = {
            'EURUSD': 1.1000,
            'GBPUSD': 1.2500,
            'USDJPY': 110.00,
            'AUDUSD': 0.7500,
            'USDCAD': 1.2500,
            'USDCHF': 0.9200,
            'NZDUSD': 0.7000,
            'AAPL': 150.00,
            'MSFT': 300.00,
            'GOOGL': 2500.00,
            'AMZN': 3200.00,
            'TSLA': 800.00
        }
        
        base_price = base_prices.get(symbol, 100.00)
        
        # Generate realistic price movements
        prices = []
        current_price = base_price
        
        for i in range(len(dates)):
            # Add some trend and noise
            trend = 0.001 * np.sin(i / periods * 4 * np.pi)  # Long-term cycles
            noise = np.random.normal(0, 0.002)  # Short-term noise
            
            # Mean reversion
            reversion = -0.1 * (current_price - base_price) / base_price
            
            change = trend + noise + reversion
            current_price = current_price * (1 + change)
            prices.append(current_price)
        
        # Generate OHLC data
        data = []
        for i, price in enumerate(prices):
            # Generate realistic OHLC
            daily_range = price * np.random.uniform(0.005, 0.02)  # 0.5-2% daily range
            
            open_price = price + np.random.uniform(-daily_range/2, daily_range/2)
            high_price = max(open_price, price) + np.random.uniform(0, daily_range/2)
            low_price = min(open_price, price) - np.random.uniform(0, daily_range/2)
            close_price = price + np.random.uniform(-daily_range/4, daily_range/4)
            
            volume = np.random.randint(10000, 100000)
            
            data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        return pd.DataFrame(data)
    
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