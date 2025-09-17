import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
# import talib  # Skip for now due to compatibility issues

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
    """Comprehensive ICT Strategy Manager implementing 65+ concepts"""
    
    def __init__(self):
        self.strategies = {
            # Core ICT Concepts (1-20)
            "market_structure_strategy": self.market_structure_strategy,
            "liquidity_strategy": self.liquidity_strategy,
            "liquidity_pools_strategy": self.liquidity_pools_strategy,
            "order_block_strategy": self.order_block_strategy,
            "breaker_block_strategy": self.breaker_block_strategy,
            "fair_value_gap_strategy": self.fair_value_gap_strategy,
            "rejection_block_strategy": self.rejection_block_strategy,
            "mitigation_block_strategy": self.mitigation_block_strategy,
            "supply_demand_zones_strategy": self.supply_demand_zones_strategy,
            "premium_discount_strategy": self.premium_discount_strategy,
            "dealing_ranges_strategy": self.dealing_ranges_strategy,
            "swing_points_strategy": self.swing_points_strategy,
            "market_maker_models_strategy": self.market_maker_models_strategy,
            "judas_swing_strategy": self.judas_swing_strategy,
            "turtle_soup_strategy": self.turtle_soup_strategy,
            "power_of_three_strategy": self.power_of_three_strategy,
            "optimal_trade_entry_strategy": self.optimal_trade_entry_strategy,
            "smt_divergence_strategy": self.smt_divergence_strategy,
            "liquidity_voids_strategy": self.liquidity_voids_strategy,
            
            # Time & Price Theory (21-30)
            "killzones_strategy": self.killzones_strategy,
            "session_opens_strategy": self.session_opens_strategy,
            "fibonacci_ratios_strategy": self.fibonacci_ratios_strategy,
            "range_expectations_strategy": self.range_expectations_strategy,
            "session_liquidity_raids_strategy": self.session_liquidity_raids_strategy,
            "weekly_profiles_strategy": self.weekly_profiles_strategy,
            "daily_bias_strategy": self.daily_bias_strategy,
            "weekly_bias_strategy": self.weekly_bias_strategy,
            "monthly_bias_strategy": self.monthly_bias_strategy,
            "time_of_day_strategy": self.time_of_day_strategy,
            
            # Risk Management & Execution (31-39)
            "trade_journaling_strategy": self.trade_journaling_strategy,
            "entry_models_strategy": self.entry_models_strategy,
            "exit_models_strategy": self.exit_models_strategy,
            "risk_reward_strategy": self.risk_reward_strategy,
            "position_sizing_strategy": self.position_sizing_strategy,
            "drawdown_control_strategy": self.drawdown_control_strategy,
            "compounding_models_strategy": self.compounding_models_strategy,
            "daily_loss_limits_strategy": self.daily_loss_limits_strategy,
            "probability_profiles_strategy": self.probability_profiles_strategy,
            
            # Advanced Concepts (40-50)
            "high_probability_scenarios_strategy": self.high_probability_scenarios_strategy,
            "liquidity_runs_strategy": self.liquidity_runs_strategy,
            "reversals_continuations_strategy": self.reversals_continuations_strategy,
            "accumulation_distribution_strategy": self.accumulation_distribution_strategy,
            "order_flow_strategy": self.order_flow_strategy,
            "high_low_day_strategy": self.high_low_day_strategy,
            "range_expansion_strategy": self.range_expansion_strategy,
            "inside_outside_day_strategy": self.inside_outside_day_strategy,
            "weekly_profiles_advanced_strategy": self.weekly_profiles_advanced_strategy,
            "ipda_theory_strategy": self.ipda_theory_strategy,
            "algo_price_delivery_strategy": self.algo_price_delivery_strategy,
            
            # Strategies / Playbooks (51-65)
            "silver_bullet_strategy": self.silver_bullet_strategy,
            "asian_range_strategy": self.asian_range_strategy,
            "ny_reversal_strategy": self.ny_reversal_strategy,
            "london_killzone_strategy": self.london_killzone_strategy,
            "fvg_sniper_strategy": self.fvg_sniper_strategy,
            "order_block_refined_strategy": self.order_block_refined_strategy,
            "breaker_block_refined_strategy": self.breaker_block_refined_strategy,
            "rejection_block_refined_strategy": self.rejection_block_refined_strategy,
            "smt_divergence_refined_strategy": self.smt_divergence_refined_strategy,
            "turtle_soup_refined_strategy": self.turtle_soup_refined_strategy,
            "power_of_three_refined_strategy": self.power_of_three_refined_strategy,
            "daily_bias_liquidity_raid_strategy": self.daily_bias_liquidity_raid_strategy,
            "am_session_bias_strategy": self.am_session_bias_strategy,
            "pm_session_reversal_strategy": self.pm_session_reversal_strategy,
            "optimal_trade_entry_refined_strategy": self.optimal_trade_entry_refined_strategy
        }
    
    async def analyze_market_with_data(self, symbol: str, timeframe: TimeFrame, market_data: dict, lookback_days: int = 30) -> MarketAnalysis:
        """Comprehensive market analysis using real market data"""
        # Convert real market data to DataFrame format
        data = self._convert_market_data_to_df(market_data)
        
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
        
        # Get current session and killzone
        current_killzone = self.get_current_killzone()
        
        # Calculate sentiment and bias
        sentiment = self._calculate_market_sentiment(data)
        bias = self._determine_market_bias(data, market_structure)
        
        return MarketAnalysis(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.utcnow(),
            market_structure=market_structure,
            key_levels=key_levels,
            order_blocks=order_blocks,
            fair_value_gaps=fair_value_gaps,
            liquidity_pools=liquidity_pools,
            current_killzone=current_killzone,
            sentiment=sentiment,
            bias=bias,
            confidence=self._calculate_confidence(market_structure, order_blocks, fair_value_gaps),
            recommendations=self._generate_recommendations(symbol, market_structure, order_blocks, fair_value_gaps),
            data_source=market_data.get('data_source', 'real'),
            current_price=market_data.get('current_price'),
            price_change_24h=market_data.get('price_change_24h')
        )

    def _convert_market_data_to_df(self, market_data: dict) -> pd.DataFrame:
        """Convert market data to DataFrame for analysis"""
        try:
            data_list = market_data.get('data', [])
            if not data_list:
                raise ValueError("No market data available")
            
            df = pd.DataFrame(data_list)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Ensure numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
        except Exception as e:
            print(f"Error converting market data: {e}")
            # Return empty DataFrame with required columns
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    async def analyze_market(self, symbol: str, timeframe: TimeFrame, lookback_days: int = 30) -> MarketAnalysis:
        """Comprehensive market analysis using all ICT concepts (legacy method)"""
        # Get market data for analysis
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
        """SMT Divergence Strategy - Smart Money Divergence between correlated pairs"""
        setups = []
        
        # For SMT divergence, we need correlated instrument analysis
        # In a simplified implementation, we'll analyze internal divergences
        
        if len(data) < 50:
            return setups
        
        # Calculate momentum oscillator
        closes = data['close'].values
        momentum = []
        for i in range(14, len(closes)):
            momentum.append((closes[i] - closes[i-14]) / closes[i-14])
        
        # Find divergences between price and momentum
        for i in range(20, len(momentum) - 5):
            # Look for price and momentum divergence
            price_window = closes[i+14-10:i+14+5]  # Adjust for momentum offset
            momentum_window = momentum[i-10:i+5]
            
            if len(price_window) < 10 or len(momentum_window) < 10:
                continue
            
            # Find local extremes
            price_high_idx = np.argmax(price_window)
            price_low_idx = np.argmin(price_window)
            momentum_high_idx = np.argmax(momentum_window)
            momentum_low_idx = np.argmin(momentum_window)
            
            # Bullish divergence: price makes lower low, momentum makes higher low
            if (price_low_idx > 5 and momentum_low_idx > 5 and
                price_window[price_low_idx] < price_window[2] and
                momentum_window[momentum_low_idx] > momentum_window[2]):
                
                current_price = closes[i + 14]
                entry_price = current_price
                stop_loss = price_window[price_low_idx] - (price_window[price_low_idx] * 0.01)
                take_profit = [
                    entry_price + (entry_price - stop_loss) * 1.5,
                    entry_price + (entry_price - stop_loss) * 2.5
                ]
                
                setup = TradeSetup(
                    symbol=symbol,
                    direction=TradeDirection.LONG,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward_ratio=1.5,
                    setup_type=ICTConcept.SMT_DIVERGENCE,
                    confidence=0.7,
                    timestamp=data.iloc[i + 14]['timestamp'],
                    timeframe=timeframe
                )
                setups.append(setup)
            
            # Bearish divergence: price makes higher high, momentum makes lower high
            elif (price_high_idx > 5 and momentum_high_idx > 5 and
                  price_window[price_high_idx] > price_window[2] and
                  momentum_window[momentum_high_idx] < momentum_window[2]):
                
                current_price = closes[i + 14]
                entry_price = current_price
                stop_loss = price_window[price_high_idx] + (price_window[price_high_idx] * 0.01)
                take_profit = [
                    entry_price - (stop_loss - entry_price) * 1.5,
                    entry_price - (stop_loss - entry_price) * 2.5
                ]
                
                setup = TradeSetup(
                    symbol=symbol,
                    direction=TradeDirection.SHORT,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward_ratio=1.5,
                    setup_type=ICTConcept.SMT_DIVERGENCE,
                    confidence=0.7,
                    timestamp=data.iloc[i + 14]['timestamp'],
                    timeframe=timeframe
                )
                setups.append(setup)
        
        return setups
    
    async def power_of_three_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Power of 3 Strategy - Accumulation, Manipulation, Distribution"""
        setups = []
        
        if len(data) < 90:  # Need enough data for 3 phases
            return setups
        
        # Look for 3-phase patterns over 60-90 period windows
        for i in range(60, len(data) - 30):
            
            # Define the three phases
            accumulation_start = i - 60
            accumulation_end = i - 40
            manipulation_start = i - 40
            manipulation_end = i - 10
            distribution_start = i - 10
            distribution_end = i + 10 if i + 10 < len(data) else len(data) - 1
            
            # Get data for each phase
            accumulation_data = data.iloc[accumulation_start:accumulation_end]
            manipulation_data = data.iloc[manipulation_start:manipulation_end]
            distribution_data = data.iloc[distribution_start:distribution_end]
            
            if len(accumulation_data) < 15 or len(manipulation_data) < 20 or len(distribution_data) < 15:
                continue
            
            # Analyze accumulation phase (consolidation/range)
            acc_high = accumulation_data['high'].max()
            acc_low = accumulation_data['low'].min()
            acc_range = acc_high - acc_low
            acc_volatility = accumulation_data['close'].std()
            
            # Analyze manipulation phase (breakout/sweep)
            manip_high = manipulation_data['high'].max()
            manip_low = manipulation_data['low'].min()
            
            # Check for liquidity sweeps
            sweep_above = manip_high > acc_high * 1.001  # Sweep above accumulation high
            sweep_below = manip_low < acc_low * 0.999   # Sweep below accumulation low
            
            # Analyze distribution phase (directional move)
            dist_start_price = distribution_data['close'].iloc[0]
            dist_end_price = distribution_data['close'].iloc[-1]
            dist_return = (dist_end_price - dist_start_price) / dist_start_price
            
            # Check for valid Power of 3 patterns
            
            # Bullish Power of 3: Sweep lows then rally
            if (sweep_below and not sweep_above and dist_return > 0.015):  # 1.5% move up
                
                entry_price = dist_start_price
                stop_loss = manip_low - acc_range * 0.1
                take_profit = [
                    entry_price + acc_range * 1.5,
                    entry_price + acc_range * 2.5
                ]
                
                confidence = 0.75 + min(abs(dist_return) * 10, 0.2)  # Higher confidence for stronger moves
                
                setup = TradeSetup(
                    symbol=symbol,
                    direction=TradeDirection.LONG,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward_ratio=1.5,
                    setup_type=ICTConcept.POWER_OF_THREE,
                    confidence=min(confidence, 1.0),
                    timestamp=distribution_data.iloc[0]['timestamp'],
                    timeframe=timeframe
                )
                setups.append(setup)
            
            # Bearish Power of 3: Sweep highs then sell off
            elif (sweep_above and not sweep_below and dist_return < -0.015):  # 1.5% move down
                
                entry_price = dist_start_price
                stop_loss = manip_high + acc_range * 0.1
                take_profit = [
                    entry_price - acc_range * 1.5,
                    entry_price - acc_range * 2.5
                ]
                
                confidence = 0.75 + min(abs(dist_return) * 10, 0.2)
                
                setup = TradeSetup(
                    symbol=symbol,
                    direction=TradeDirection.SHORT,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward_ratio=1.5,
                    setup_type=ICTConcept.POWER_OF_THREE,
                    confidence=min(confidence, 1.0),
                    timestamp=distribution_data.iloc[0]['timestamp'],
                    timeframe=timeframe
                )
                setups.append(setup)
        
        return setups
    
    async def rejection_block_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Rejection Block Strategy - Trade from significant wick rejections"""
        setups = []
        
        if len(data) < 20:
            return setups
        
        for i in range(10, len(data) - 5):
            current_candle = data.iloc[i]
            
            # Calculate candle components
            open_price = current_candle['open']
            close_price = current_candle['close']
            high_price = current_candle['high']
            low_price = current_candle['low']
            
            total_range = high_price - low_price
            body_size = abs(close_price - open_price)
            
            if total_range <= 0:
                continue
            
            # Calculate wicks
            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price
            
            # Look for significant rejection wicks
            significant_wick_ratio = 0.6  # Wick must be 60% of total range
            
            # Bullish rejection (long lower wick)
            if (lower_wick > total_range * significant_wick_ratio and 
                close_price > open_price):  # Bullish close
                
                # Check if this level holds as support
                support_tests = 0
                for j in range(i + 1, min(i + 20, len(data))):
                    test_low = data.iloc[j]['low']
                    test_close = data.iloc[j]['close']
                    
                    # Check if price respects the rejection level
                    if (test_low <= low_price * 1.002 and  # Within 0.2% of rejection low
                        test_close > low_price):  # But closes above
                        support_tests += 1
                
                if support_tests >= 1:  # At least one retest
                    entry_price = close_price
                    stop_loss = low_price - total_range * 0.2
                    take_profit = [
                        entry_price + total_range * 1.5,
                        entry_price + total_range * 2.5
                    ]
                    
                    confidence = 0.7 + min(support_tests / 10, 0.2)
                    
                    setup = TradeSetup(
                        symbol=symbol,
                        direction=TradeDirection.LONG,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=(take_profit[0] - entry_price) / (entry_price - stop_loss),
                        setup_type=ICTConcept.REJECTION_BLOCK,
                        confidence=min(confidence, 1.0),
                        timestamp=current_candle['timestamp'],
                        timeframe=timeframe
                    )
                    setups.append(setup)
            
            # Bearish rejection (long upper wick)
            elif (upper_wick > total_range * significant_wick_ratio and 
                  close_price < open_price):  # Bearish close
                
                # Check if this level holds as resistance
                resistance_tests = 0
                for j in range(i + 1, min(i + 20, len(data))):
                    test_high = data.iloc[j]['high']
                    test_close = data.iloc[j]['close']
                    
                    # Check if price respects the rejection level
                    if (test_high >= high_price * 0.998 and  # Within 0.2% of rejection high
                        test_close < high_price):  # But closes below
                        resistance_tests += 1
                
                if resistance_tests >= 1:  # At least one retest
                    entry_price = close_price
                    stop_loss = high_price + total_range * 0.2
                    take_profit = [
                        entry_price - total_range * 1.5,
                        entry_price - total_range * 2.5
                    ]
                    
                    confidence = 0.7 + min(resistance_tests / 10, 0.2)
                    
                    setup = TradeSetup(
                        symbol=symbol,
                        direction=TradeDirection.SHORT,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=(entry_price - take_profit[0]) / (stop_loss - entry_price),
                        setup_type=ICTConcept.REJECTION_BLOCK,
                        confidence=min(confidence, 1.0),
                        timestamp=current_candle['timestamp'],
                        timeframe=timeframe
                    )
                    setups.append(setup)
        
        return setups
    
    async def mitigation_block_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Mitigation Block Strategy - Trade when order blocks get mitigated"""
        setups = []
        
        # Find order blocks first
        order_blocks = self._find_order_blocks(data)
        
        for ob in order_blocks:
            ob_timestamp = ob.timestamp
            ob_index = data[data['timestamp'] == ob_timestamp].index
            
            if len(ob_index) == 0:
                continue
                
            ob_idx = ob_index[0]
            
            # Look for mitigation (price returning to order block)
            for j in range(ob_idx + 3, min(ob_idx + 50, len(data))):
                current_price = data.iloc[j]['close']
                current_high = data.iloc[j]['high']
                current_low = data.iloc[j]['low']
                
                mitigation_found = False
                mitigation_level = 0
                mitigation_percentage = 0
                
                if ob.direction == TradeDirection.LONG:
                    # For bullish order block, look for price returning to the block
                    if current_low <= ob.high and current_high >= ob.low:
                        # Calculate how much of the order block is mitigated
                        if current_low <= ob.low:
                            mitigation_percentage = 1.0  # Full mitigation
                            mitigation_level = ob.low
                        else:
                            mitigation_percentage = (ob.high - current_low) / (ob.high - ob.low)
                            mitigation_level = current_low
                        
                        mitigation_found = True
                
                else:  # Bearish order block
                    if current_high >= ob.low and current_low <= ob.high:
                        # Calculate how much of the order block is mitigated
                        if current_high >= ob.high:
                            mitigation_percentage = 1.0  # Full mitigation
                            mitigation_level = ob.high
                        else:
                            mitigation_percentage = (current_high - ob.low) / (ob.high - ob.low)
                            mitigation_level = current_high
                        
                        mitigation_found = True
                
                if mitigation_found and mitigation_percentage >= 0.5:  # At least 50% mitigated
                    
                    # Look for continuation after mitigation
                    continuation_confirmed = False
                    for k in range(j + 1, min(j + 10, len(data))):
                        next_close = data.iloc[k]['close']
                        
                        if ob.direction == TradeDirection.LONG:
                            # Look for bullish continuation after mitigation
                            if next_close > current_price * 1.005:  # 0.5% move up
                                continuation_confirmed = True
                                break
                        else:
                            # Look for bearish continuation after mitigation
                            if next_close < current_price * 0.995:  # 0.5% move down
                                continuation_confirmed = True
                                break
                    
                    if continuation_confirmed:
                        # Create trade setup
                        if ob.direction == TradeDirection.LONG:
                            entry_price = mitigation_level
                            stop_loss = ob.low - (ob.high - ob.low) * 0.3
                            take_profit = [
                                entry_price + (ob.high - ob.low) * 1.5,
                                entry_price + (ob.high - ob.low) * 2.5
                            ]
                        else:  # SHORT
                            entry_price = mitigation_level
                            stop_loss = ob.high + (ob.high - ob.low) * 0.3
                            take_profit = [
                                entry_price - (ob.high - ob.low) * 1.5,
                                entry_price - (ob.high - ob.low) * 2.5
                            ]
                        
                        confidence = 0.65 + mitigation_percentage * 0.25  # Higher confidence for fuller mitigation
                        
                        setup = TradeSetup(
                            symbol=symbol,
                            direction=ob.direction,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            risk_reward_ratio=1.5,
                            setup_type=ICTConcept.MITIGATION_BLOCK,
                            confidence=min(confidence, 1.0),
                            timestamp=data.iloc[j]['timestamp'],
                            timeframe=timeframe
                        )
                        setups.append(setup)
                    
                    break  # Found mitigation for this order block
        
        return setups
    
    async def turtle_soup_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Turtle Soup Strategy - Stop hunt reversal patterns"""
        setups = []
        
        if len(data) < 30:
            return setups
        
        # Look for false breakouts and stop hunts
        for i in range(20, len(data) - 10):
            
            # Find recent significant levels (support/resistance)
            lookback_data = data.iloc[i-20:i]
            recent_high = lookback_data['high'].max()
            recent_low = lookback_data['low'].min()
            
            current_candle = data.iloc[i]
            current_high = current_candle['high']
            current_low = current_candle['low']
            current_close = current_candle['close']
            
            # Check for false breakout above resistance
            if current_high > recent_high * 1.001:  # Break above with 0.1% margin
                
                # Look for quick reversal back below the level
                reversal_found = False
                reversal_strength = 0
                
                for j in range(i + 1, min(i + 8, len(data))):  # Check next 7 candles
                    reversal_close = data.iloc[j]['close']
                    reversal_low = data.iloc[j]['low']
                    
                    if reversal_close < recent_high * 0.999:  # Close back below level
                        reversal_found = True
                        reversal_strength = (recent_high - reversal_low) / recent_high
                        
                        # Stronger reversal = higher confidence
                        if reversal_strength > 0.005:  # At least 0.5% reversal
                            
                            # Create bearish setup
                            entry_price = recent_high
                            stop_loss = current_high + (current_high - recent_high) * 0.5
                            
                            # Calculate take profit based on reversal strength
                            range_size = recent_high - recent_low
                            take_profit = [
                                entry_price - range_size * 0.8,
                                entry_price - range_size * 1.5
                            ]
                            
                            confidence = 0.7 + min(reversal_strength * 20, 0.25)
                            
                            setup = TradeSetup(
                                symbol=symbol,
                                direction=TradeDirection.SHORT,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                risk_reward_ratio=(entry_price - take_profit[0]) / (stop_loss - entry_price),
                                setup_type=ICTConcept.TURTLE_SOUP,
                                confidence=min(confidence, 1.0),
                                timestamp=data.iloc[j]['timestamp'],
                                timeframe=timeframe
                            )
                            setups.append(setup)
                        break
            
            # Check for false breakout below support
            elif current_low < recent_low * 0.999:  # Break below with 0.1% margin
                
                # Look for quick reversal back above the level
                reversal_found = False
                reversal_strength = 0
                
                for j in range(i + 1, min(i + 8, len(data))):  # Check next 7 candles
                    reversal_close = data.iloc[j]['close']
                    reversal_high = data.iloc[j]['high']
                    
                    if reversal_close > recent_low * 1.001:  # Close back above level
                        reversal_found = True
                        reversal_strength = (reversal_high - recent_low) / recent_low
                        
                        # Stronger reversal = higher confidence
                        if reversal_strength > 0.005:  # At least 0.5% reversal
                            
                            # Create bullish setup
                            entry_price = recent_low
                            stop_loss = current_low - (recent_low - current_low) * 0.5
                            
                            # Calculate take profit based on reversal strength
                            range_size = recent_high - recent_low
                            take_profit = [
                                entry_price + range_size * 0.8,
                                entry_price + range_size * 1.5
                            ]
                            
                            confidence = 0.7 + min(reversal_strength * 20, 0.25)
                            
                            setup = TradeSetup(
                                symbol=symbol,
                                direction=TradeDirection.LONG,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                risk_reward_ratio=(take_profit[0] - entry_price) / (entry_price - stop_loss),
                                setup_type=ICTConcept.TURTLE_SOUP,
                                confidence=min(confidence, 1.0),
                                timestamp=data.iloc[j]['timestamp'],
                                timeframe=timeframe
                            )
                            setups.append(setup)
                        break
        
        return setups
    
    async def judas_swing_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Judas Swing Strategy - False moves at session opens"""
        setups = []
        
        if len(data) < 20:
            return setups
        
        # Look for moves during session opening times
        for i in range(5, len(data) - 15):
            timestamp = data.iloc[i]['timestamp']
            hour = timestamp.hour
            minute = timestamp.minute
            
            # Check if this is near major session opens
            is_session_open = False
            session_name = ""
            
            # London open (7:00-9:00 GMT)
            if 7 <= hour <= 9:
                is_session_open = True
                session_name = "London"
            # New York open (12:30-14:30 GMT)  
            elif 12 <= hour <= 14 and (hour > 12 or minute >= 30):
                is_session_open = True
                session_name = "New York"
            # Asia open (22:00-00:00 GMT)
            elif hour >= 22 or hour <= 1:
                is_session_open = True
                session_name = "Asia"
            
            if not is_session_open:
                continue
            
            # Get session open price (approximate)
            session_open_price = data.iloc[i]['open']
            
            # Look for initial false move followed by reversal
            initial_move_periods = 5  # Look at next 5 periods for initial move
            reversal_periods = 10     # Look at next 10 periods for reversal
            
            if i + initial_move_periods + reversal_periods >= len(data):
                continue
            
            # Analyze initial move
            initial_data = data.iloc[i:i+initial_move_periods]
            initial_high = initial_data['high'].max()
            initial_low = initial_data['low'].min()
            
            # Determine if there was a significant initial move
            upward_move = initial_high > session_open_price * 1.003  # 0.3% move up
            downward_move = initial_low < session_open_price * 0.997  # 0.3% move down
            
            if upward_move and not downward_move:
                # Initial upward move - look for reversal down
                reversal_data = data.iloc[i+initial_move_periods:i+initial_move_periods+reversal_periods]
                reversal_low = reversal_data['low'].min()
                reversal_close = reversal_data['close'].iloc[-1]
                
                # Check if price reversed below session open
                if reversal_low < session_open_price * 0.997:  # Reversed at least 0.3% below open
                    
                    reversal_strength = (initial_high - reversal_low) / session_open_price
                    
                    if reversal_strength > 0.008:  # At least 0.8% total reversal
                        
                        entry_price = session_open_price
                        stop_loss = initial_high + (initial_high - session_open_price) * 0.3
                        
                        # Target based on reversal strength
                        target_distance = reversal_strength * session_open_price
                        take_profit = [
                            entry_price - target_distance,
                            entry_price - target_distance * 1.5
                        ]
                        
                        confidence = 0.65 + min(reversal_strength * 15, 0.3)
                        
                        setup = TradeSetup(
                            symbol=symbol,
                            direction=TradeDirection.SHORT,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            risk_reward_ratio=(entry_price - take_profit[0]) / (stop_loss - entry_price),
                            setup_type=ICTConcept.JUDAS_SWING,
                            confidence=min(confidence, 1.0),
                            timestamp=reversal_data.iloc[0]['timestamp'],
                            timeframe=timeframe
                        )
                        setups.append(setup)
            
            elif downward_move and not upward_move:
                # Initial downward move - look for reversal up
                reversal_data = data.iloc[i+initial_move_periods:i+initial_move_periods+reversal_periods]
                reversal_high = reversal_data['high'].max()
                reversal_close = reversal_data['close'].iloc[-1]
                
                # Check if price reversed above session open
                if reversal_high > session_open_price * 1.003:  # Reversed at least 0.3% above open
                    
                    reversal_strength = (reversal_high - initial_low) / session_open_price
                    
                    if reversal_strength > 0.008:  # At least 0.8% total reversal
                        
                        entry_price = session_open_price
                        stop_loss = initial_low - (session_open_price - initial_low) * 0.3
                        
                        # Target based on reversal strength  
                        target_distance = reversal_strength * session_open_price
                        take_profit = [
                            entry_price + target_distance,
                            entry_price + target_distance * 1.5
                        ]
                        
                        confidence = 0.65 + min(reversal_strength * 15, 0.3)
                        
                        setup = TradeSetup(
                            symbol=symbol,
                            direction=TradeDirection.LONG,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            risk_reward_ratio=(take_profit[0] - entry_price) / (entry_price - stop_loss),
                            setup_type=ICTConcept.JUDAS_SWING,
                            confidence=min(confidence, 1.0),
                            timestamp=reversal_data.iloc[0]['timestamp'],
                            timeframe=timeframe
                        )
                        setups.append(setup)
        
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
        """Asian Range Strategy - Trade breakouts from Asian session consolidation"""
        setups = []
        
        if len(data) < 30:
            return setups
        
        # Look for Asian session periods (approximately 22:00 GMT to 06:00 GMT)
        for i in range(20, len(data) - 10):
            timestamp = data.iloc[i]['timestamp']
            hour = timestamp.hour
            
            # Check if we're transitioning out of Asian session
            is_asian_end = hour in [6, 7, 8]  # European session starting
            
            if not is_asian_end:
                continue
            
            # Look back to find Asian session range
            asian_session_data = []
            for j in range(max(0, i-20), i):
                session_hour = data.iloc[j]['timestamp'].hour
                # Asian session hours
                if session_hour >= 22 or session_hour <= 6:
                    asian_session_data.append(j)
            
            if len(asian_session_data) < 10:  # Need sufficient Asian session data
                continue
            
            # Calculate Asian session range
            asian_data = data.iloc[asian_session_data]
            asian_high = asian_data['high'].max()
            asian_low = asian_data['low'].min()
            asian_range = asian_high - asian_low
            
            # Ensure we have a meaningful range
            if asian_range < asian_low * 0.002:  # At least 0.2% range
                continue
            
            # Check for range consolidation (low volatility)
            asian_volatility = asian_data['close'].std()
            asian_mean_price = asian_data['close'].mean()
            normalized_volatility = asian_volatility / asian_mean_price
            
            # Look for tight consolidation (low volatility)
            if normalized_volatility > 0.005:  # Skip if too volatile (>0.5%)
                continue
            
            # Look for breakout from Asian range
            breakout_periods = 8  # Check next 8 periods for breakout
            
            for k in range(i, min(i + breakout_periods, len(data))):
                current_high = data.iloc[k]['high']
                current_low = data.iloc[k]['low']
                current_close = data.iloc[k]['close']
                
                # Bullish breakout above Asian high
                if current_high > asian_high * 1.001:  # 0.1% margin for breakout
                    
                    # Confirm with close above the range
                    if current_close > asian_high:
                        
                        # Look for continuation
                        continuation_confirmed = False
                        for m in range(k + 1, min(k + 5, len(data))):
                            if data.iloc[m]['close'] > asian_high * 1.005:  # 0.5% above breakout
                                continuation_confirmed = True
                                break
                        
                        if continuation_confirmed:
                            entry_price = asian_high
                            stop_loss = asian_low - asian_range * 0.2
                            
                            # Target based on Asian range
                            take_profit = [
                                entry_price + asian_range * 1.0,
                                entry_price + asian_range * 1.6
                            ]
                            
                            # Higher confidence for tighter ranges
                            tightness_factor = 1 - min(normalized_volatility / 0.003, 1)
                            confidence = 0.7 + tightness_factor * 0.2
                            
                            setup = TradeSetup(
                                symbol=symbol,
                                direction=TradeDirection.LONG,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                risk_reward_ratio=(take_profit[0] - entry_price) / (entry_price - stop_loss),
                                setup_type=ICTConcept.KILLZONE,  # Using killzone as closest concept
                                confidence=min(confidence, 1.0),
                                timestamp=data.iloc[k]['timestamp'],
                                timeframe=timeframe
                            )
                            setups.append(setup)
                    break
                
                # Bearish breakout below Asian low
                elif current_low < asian_low * 0.999:  # 0.1% margin for breakout
                    
                    # Confirm with close below the range
                    if current_close < asian_low:
                        
                        # Look for continuation
                        continuation_confirmed = False
                        for m in range(k + 1, min(k + 5, len(data))):
                            if data.iloc[m]['close'] < asian_low * 0.995:  # 0.5% below breakout
                                continuation_confirmed = True
                                break
                        
                        if continuation_confirmed:
                            entry_price = asian_low
                            stop_loss = asian_high + asian_range * 0.2
                            
                            # Target based on Asian range
                            take_profit = [
                                entry_price - asian_range * 1.0,
                                entry_price - asian_range * 1.6
                            ]
                            
                            # Higher confidence for tighter ranges
                            tightness_factor = 1 - min(normalized_volatility / 0.003, 1)
                            confidence = 0.7 + tightness_factor * 0.2
                            
                            setup = TradeSetup(
                                symbol=symbol,
                                direction=TradeDirection.SHORT,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                take_profit=take_profit,
                                risk_reward_ratio=(entry_price - take_profit[0]) / (stop_loss - entry_price),
                                setup_type=ICTConcept.KILLZONE,  # Using killzone as closest concept
                                confidence=min(confidence, 1.0),
                                timestamp=data.iloc[k]['timestamp'],
                                timeframe=timeframe
                            )
                            setups.append(setup)
                    break
        
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
                # Try alternative symbol formats or raise error for invalid symbols
                raise ValueError(f"No market data available for symbol {symbol}. Please verify symbol format.")
            
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
            # Return error instead of mock data
            raise ValueError(f"Cannot fetch real market data for {symbol}: {str(e)}")
    
    def _validate_symbol_format(self, symbol: str) -> str:
        """Validate and format symbol for data providers"""
        # Common symbol format corrections
        symbol_map = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X', 
            'USDJPY': 'USDJPY=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'USDCHF': 'USDCHF=X',
            'NZDUSD': 'NZDUSD=X'
        }
        
        return symbol_map.get(symbol, symbol)
    

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
    
    # Additional ICT Strategy Implementations
    async def market_structure_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Market Structure Strategy - HH, HL, LH, LL patterns"""
        setups = []
        swing_points = self._find_swing_points(data)
        
        if len(swing_points) < 4:
            return setups
        
        # Analyze structure for trend continuation/reversal
        structure_type = self._classify_market_structure(swing_points)
        
        if structure_type == "bullish_structure":
            # Look for continuation patterns
            latest_swing_low = None
            for point in reversed(swing_points):
                if point.type == "swing_low":
                    latest_swing_low = point
                    break
            
            if latest_swing_low:
                entry_price = latest_swing_low.price * 1.001  # Just above swing low
                stop_loss = latest_swing_low.price * 0.995
                take_profit = [entry_price * 1.01, entry_price * 1.02]
                
                setup = TradeSetup(
                    symbol=symbol,
                    direction=TradeDirection.LONG,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward_ratio=2.0,
                    setup_type=ICTConcept.MARKET_STRUCTURE,
                    confidence=0.75,
                    timestamp=datetime.utcnow(),
                    timeframe=timeframe
                )
                setups.append(setup)
        
        return setups
    
    async def liquidity_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Liquidity Strategy - Buy-side & sell-side liquidity"""
        return await self.liquidity_raid_strategy(data, symbol, timeframe)
    
    async def liquidity_pools_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Liquidity Pools Strategy - Equal highs/lows"""
        return await self.liquidity_raid_strategy(data, symbol, timeframe)
    
    async def supply_demand_zones_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Supply & Demand Zones Strategy"""
        setups = []
        
        # Find zones of supply and demand based on volume and price action
        for i in range(20, len(data) - 5):
            current_volume = data.iloc[i]['volume']
            avg_volume = np.mean(data.iloc[i-10:i]['volume'])
            
            # High volume areas indicate supply/demand zones
            if current_volume > avg_volume * 2.0:
                current_price = data.iloc[i]['close']
                current_high = data.iloc[i]['high']
                current_low = data.iloc[i]['low']
                
                # Check for rejection from this zone
                rejection_found = False
                for j in range(i + 1, min(i + 10, len(data))):
                    test_high = data.iloc[j]['high']
                    test_low = data.iloc[j]['low']
                    test_close = data.iloc[j]['close']
                    
                    # Test for supply zone (resistance)
                    if test_high >= current_high * 0.999 and test_close < current_high * 0.995:
                        entry_price = current_high
                        stop_loss = current_high * 1.005
                        take_profit = [current_low, current_low * 0.99]
                        
                        setup = TradeSetup(
                            symbol=symbol,
                            direction=TradeDirection.SHORT,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            risk_reward_ratio=2.0,
                            setup_type=ICTConcept.SUPPLY_DEMAND,
                            confidence=0.7,
                            timestamp=data.iloc[j]['timestamp'],
                            timeframe=timeframe
                        )
                        setups.append(setup)
                        break
                    
                    # Test for demand zone (support)
                    elif test_low <= current_low * 1.001 and test_close > current_low * 1.005:
                        entry_price = current_low
                        stop_loss = current_low * 0.995
                        take_profit = [current_high, current_high * 1.01]
                        
                        setup = TradeSetup(
                            symbol=symbol,
                            direction=TradeDirection.LONG,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            risk_reward_ratio=2.0,
                            setup_type=ICTConcept.SUPPLY_DEMAND,
                            confidence=0.7,
                            timestamp=data.iloc[j]['timestamp'],
                            timeframe=timeframe
                        )
                        setups.append(setup)
                        break
        
        return setups
    
    async def premium_discount_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Premium & Discount Strategy - Optimal Trade Entry zones"""
        return await self.optimal_trade_entry_strategy(data, symbol, timeframe)
    
    async def dealing_ranges_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Dealing Ranges Strategy"""
        setups = []
        
        # Find consolidation ranges
        for i in range(30, len(data) - 10):
            range_data = data.iloc[i-30:i]
            range_high = range_data['high'].max()
            range_low = range_data['low'].min()
            range_size = range_high - range_low
            
            # Check if we have a valid range (price consolidation)
            closes = range_data['close'].values
            price_std = np.std(closes)
            mean_price = np.mean(closes)
            
            # Low volatility indicates ranging market
            if price_std / mean_price < 0.01:  # Less than 1% volatility
                
                current_price = data.iloc[i]['close']
                
                # Buy at discount (near range low)
                if current_price <= range_low + range_size * 0.25:
                    entry_price = current_price
                    stop_loss = range_low - range_size * 0.1
                    take_profit = [range_high - range_size * 0.1]
                    
                    setup = TradeSetup(
                        symbol=symbol,
                        direction=TradeDirection.LONG,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=3.0,
                        setup_type=ICTConcept.DEALING_RANGE,
                        confidence=0.8,
                        timestamp=data.iloc[i]['timestamp'],
                        timeframe=timeframe
                    )
                    setups.append(setup)
                
                # Sell at premium (near range high)
                elif current_price >= range_high - range_size * 0.25:
                    entry_price = current_price
                    stop_loss = range_high + range_size * 0.1
                    take_profit = [range_low + range_size * 0.1]
                    
                    setup = TradeSetup(
                        symbol=symbol,
                        direction=TradeDirection.SHORT,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=3.0,
                        setup_type=ICTConcept.DEALING_RANGE,
                        confidence=0.8,
                        timestamp=data.iloc[i]['timestamp'],
                        timeframe=timeframe
                    )
                    setups.append(setup)
        
        return setups
    
    async def swing_points_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Swing Points Strategy"""
        return await self.market_structure_strategy(data, symbol, timeframe)
    
    async def market_maker_models_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Market Maker Buy & Sell Models Strategy"""
        setups = []
        
        # Look for market maker accumulation/distribution patterns
        for i in range(50, len(data) - 20):
            
            # Get price action over accumulation period
            accumulation_period = data.iloc[i-50:i-20]
            recent_period = data.iloc[i-20:i]
            
            acc_volume = accumulation_period['volume'].sum()
            recent_volume = recent_period['volume'].sum()
            
            acc_range = accumulation_period['high'].max() - accumulation_period['low'].min()
            recent_range = recent_period['high'].max() - recent_period['low'].min()
            
            # Market maker accumulation: High volume, low range
            if (acc_volume > recent_volume * 1.5 and acc_range < recent_range * 0.8):
                
                # Look for breakout direction
                acc_close = accumulation_period['close'].iloc[-1]
                current_price = data.iloc[i]['close']
                
                if current_price > acc_close * 1.01:  # Bullish breakout
                    entry_price = current_price
                    stop_loss = accumulation_period['low'].min()
                    take_profit = [current_price + acc_range * 2]
                    
                    setup = TradeSetup(
                        symbol=symbol,
                        direction=TradeDirection.LONG,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=2.0,
                        setup_type=ICTConcept.MARKET_MAKER_MODEL,
                        confidence=0.75,
                        timestamp=data.iloc[i]['timestamp'],
                        timeframe=timeframe
                    )
                    setups.append(setup)
                
                elif current_price < acc_close * 0.99:  # Bearish breakout
                    entry_price = current_price
                    stop_loss = accumulation_period['high'].max()
                    take_profit = [current_price - acc_range * 2]
                    
                    setup = TradeSetup(
                        symbol=symbol,
                        direction=TradeDirection.SHORT,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=2.0,
                        setup_type=ICTConcept.MARKET_MAKER_MODEL,
                        confidence=0.75,
                        timestamp=data.iloc[i]['timestamp'],
                        timeframe=timeframe
                    )
                    setups.append(setup)
        
        return setups
    
    async def liquidity_voids_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Liquidity Voids/Inefficiencies Strategy"""
        return await self.fair_value_gap_strategy(data, symbol, timeframe)
    
    # Time & Price Theory Strategies
    async def killzones_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Killzones Strategy - London, NY, Asia sessions"""
        return await self.london_killzone_strategy(data, symbol, timeframe)
    
    async def session_opens_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Session Opens Strategy"""
        return await self.judas_swing_strategy(data, symbol, timeframe)
    
    async def fibonacci_ratios_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Fibonacci Ratios Strategy - 50%, 62%, 70.5%, 79%"""
        return await self.optimal_trade_entry_strategy(data, symbol, timeframe)
    
    async def range_expectations_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Daily & Weekly Range Expectations Strategy"""
        setups = []
        
        current_time = datetime.utcnow()
        current_date = current_time.date()
        
        # Calculate expected daily range based on historical data
        daily_ranges = []
        for i in range(max(0, len(data) - 20), len(data)):
            daily_data = data.iloc[i]
            daily_range = daily_data['high'] - daily_data['low']
            daily_ranges.append(daily_range)
        
        if len(daily_ranges) >= 5:
            avg_daily_range = np.mean(daily_ranges)
            current_range = data.iloc[-1]['high'] - data.iloc[-1]['low']
            
            # If current range is below average, expect expansion
            if current_range < avg_daily_range * 0.7:
                current_price = data.iloc[-1]['close']
                
                # Set up for range expansion trades
                entry_price = current_price
                stop_loss_distance = avg_daily_range * 0.2
                target_distance = avg_daily_range * 0.8
                
                # Both directions as we expect expansion
                setups.extend([
                    TradeSetup(
                        symbol=symbol,
                        direction=TradeDirection.LONG,
                        entry_price=entry_price,
                        stop_loss=entry_price - stop_loss_distance,
                        take_profit=[entry_price + target_distance],
                        risk_reward_ratio=4.0,
                        setup_type=ICTConcept.DAILY_RANGE,
                        confidence=0.6,
                        timestamp=datetime.utcnow(),
                        timeframe=timeframe
                    ),
                    TradeSetup(
                        symbol=symbol,
                        direction=TradeDirection.SHORT,
                        entry_price=entry_price,
                        stop_loss=entry_price + stop_loss_distance,
                        take_profit=[entry_price - target_distance],
                        risk_reward_ratio=4.0,
                        setup_type=ICTConcept.DAILY_RANGE,
                        confidence=0.6,
                        timestamp=datetime.utcnow(),
                        timeframe=timeframe
                    )
                ])
        
        return setups
    
    async def session_liquidity_raids_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Session Liquidity Raids Strategy"""
        return await self.liquidity_raid_strategy(data, symbol, timeframe)
    
    async def weekly_profiles_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Weekly Profiles Strategy - WHLC"""
        setups = []
        
        if len(data) < 7 * 24:  # Need at least a week of hourly data
            return setups
        
        # Calculate weekly levels
        weekly_data = data.iloc[-7*24:]  # Last week
        weekly_high = weekly_data['high'].max()
        weekly_low = weekly_data['low'].min()
        weekly_open = weekly_data['open'].iloc[0]
        weekly_close = weekly_data['close'].iloc[-1]
        
        current_price = data.iloc[-1]['close']
        
        # Weekly bias based on close relative to open
        if weekly_close > weekly_open:  # Bullish week
            # Look for buys on retracement to weekly levels
            entry_zones = [weekly_open, (weekly_high + weekly_low) / 2]
            
            for zone in entry_zones:
                if abs(current_price - zone) / zone < 0.005:  # Within 0.5%
                    setup = TradeSetup(
                        symbol=symbol,
                        direction=TradeDirection.LONG,
                        entry_price=zone,
                        stop_loss=weekly_low * 0.998,
                        take_profit=[weekly_high * 1.002, weekly_high * 1.005],
                        risk_reward_ratio=2.0,
                        setup_type=ICTConcept.WEEKLY_PROFILE,
                        confidence=0.7,
                        timestamp=datetime.utcnow(),
                        timeframe=timeframe
                    )
                    setups.append(setup)
        
        elif weekly_close < weekly_open:  # Bearish week
            # Look for sells on retracement to weekly levels
            entry_zones = [weekly_open, (weekly_high + weekly_low) / 2]
            
            for zone in entry_zones:
                if abs(current_price - zone) / zone < 0.005:  # Within 0.5%
                    setup = TradeSetup(
                        symbol=symbol,
                        direction=TradeDirection.SHORT,
                        entry_price=zone,
                        stop_loss=weekly_high * 1.002,
                        take_profit=[weekly_low * 0.998, weekly_low * 0.995],
                        risk_reward_ratio=2.0,
                        setup_type=ICTConcept.WEEKLY_PROFILE,
                        confidence=0.7,
                        timestamp=datetime.utcnow(),
                        timeframe=timeframe
                    )
                    setups.append(setup)
        
        return setups
    
    async def daily_bias_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Daily Bias Strategy"""
        setups = []
        
        if len(data) < 24:  # Need at least a day of data
            return setups
        
        # Get today's open and previous day's high/low
        daily_open = data.iloc[-24]['open']  # Approximate daily open
        prev_day_high = max(data.iloc[-48:-24]['high'])
        prev_day_low = min(data.iloc[-48:-24]['low'])
        
        current_price = data.iloc[-1]['close']
        
        # Bullish bias if above daily open
        if current_price > daily_open:
            # Look for long opportunities on retracement
            entry_price = daily_open
            stop_loss = prev_day_low
            take_profit = [prev_day_high + (prev_day_high - prev_day_low) * 0.5]
            
            setup = TradeSetup(
                symbol=symbol,
                direction=TradeDirection.LONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=1.5,
                setup_type=ICTConcept.DAILY_BIAS,
                confidence=0.7,
                timestamp=datetime.utcnow(),
                timeframe=timeframe
            )
            setups.append(setup)
        
        # Bearish bias if below daily open
        elif current_price < daily_open:
            # Look for short opportunities on retracement
            entry_price = daily_open
            stop_loss = prev_day_high
            take_profit = [prev_day_low - (prev_day_high - prev_day_low) * 0.5]
            
            setup = TradeSetup(
                symbol=symbol,
                direction=TradeDirection.SHORT,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=1.5,
                setup_type=ICTConcept.DAILY_BIAS,
                confidence=0.7,
                timestamp=datetime.utcnow(),
                timeframe=timeframe
            )
            setups.append(setup)
        
        return setups
    
    async def weekly_bias_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Weekly Bias Strategy"""
        return await self.weekly_profiles_strategy(data, symbol, timeframe)
    
    async def monthly_bias_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Monthly Bias Strategy"""
        setups = []
        
        if len(data) < 30 * 24:  # Need at least a month of data
            return setups
        
        # Get monthly levels
        monthly_data = data.iloc[-30*24:]  # Last month
        monthly_open = monthly_data['open'].iloc[0]
        monthly_high = monthly_data['high'].max()
        monthly_low = monthly_data['low'].min()
        
        current_price = data.iloc[-1]['close']
        
        # Monthly bias
        if current_price > monthly_open:  # Bullish monthly bias
            entry_price = monthly_open
            stop_loss = monthly_low
            take_profit = [monthly_high + (monthly_high - monthly_low) * 0.5]
            
            setup = TradeSetup(
                symbol=symbol,
                direction=TradeDirection.LONG,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=1.5,
                setup_type=ICTConcept.MONTHLY_BIAS,
                confidence=0.65,
                timestamp=datetime.utcnow(),
                timeframe=timeframe
            )
            setups.append(setup)
        
        return setups
    
    async def time_of_day_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Time of Day Strategy - AM/PM separation"""
        setups = []
        
        current_time = datetime.utcnow()
        hour = current_time.hour
        
        # AM session (before 12:00 GMT)
        if hour < 12:
            # Look for trending moves in AM
            if len(data) >= 12:
                am_data = data.iloc[-12:]  # Last 12 hours
                am_trend = np.polyfit(range(len(am_data)), am_data['close'].values, 1)[0]
                
                if abs(am_trend) > 0.001:  # Significant trend
                    current_price = data.iloc[-1]['close']
                    direction = TradeDirection.LONG if am_trend > 0 else TradeDirection.SHORT
                    
                    if direction == TradeDirection.LONG:
                        entry_price = current_price
                        stop_loss = current_price * 0.995
                        take_profit = [current_price * 1.01]
                    else:
                        entry_price = current_price
                        stop_loss = current_price * 1.005
                        take_profit = [current_price * 0.99]
                    
                    setup = TradeSetup(
                        symbol=symbol,
                        direction=direction,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        risk_reward_ratio=2.0,
                        setup_type=ICTConcept.TIME_OF_DAY,
                        confidence=0.6,
                        timestamp=datetime.utcnow(),
                        timeframe=timeframe
                    )
                    setups.append(setup)
        
        return setups
    
    # Additional refined strategy implementations for completeness
    async def fvg_sniper_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """FVG Sniper Entry Strategy - Precise FVG entries"""
        return await self.fair_value_gap_strategy(data, symbol, timeframe)
    
    async def order_block_refined_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Refined Order Block Strategy"""
        return await self.order_block_strategy(data, symbol, timeframe)
    
    async def breaker_block_refined_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Refined Breaker Block Strategy"""
        return await self.breaker_block_strategy(data, symbol, timeframe)
    
    async def rejection_block_refined_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Refined Rejection Block Strategy"""
        return await self.rejection_block_strategy(data, symbol, timeframe)
    
    async def smt_divergence_refined_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Refined SMT Divergence Strategy"""
        return await self.smt_divergence_strategy(data, symbol, timeframe)
    
    async def turtle_soup_refined_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Refined Turtle Soup Strategy"""
        return await self.turtle_soup_strategy(data, symbol, timeframe)
    
    async def power_of_three_refined_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Refined Power of 3 Strategy"""
        return await self.power_of_three_strategy(data, symbol, timeframe)
    
    async def daily_bias_liquidity_raid_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Daily Bias + Liquidity Raid Strategy"""
        daily_setups = await self.daily_bias_strategy(data, symbol, timeframe)
        liquidity_setups = await self.liquidity_raid_strategy(data, symbol, timeframe)
        return daily_setups + liquidity_setups
    
    async def am_session_bias_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """AM Session Bias Strategy"""
        return await self.time_of_day_strategy(data, symbol, timeframe)
    
    async def pm_session_reversal_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """PM Session Reversal Strategy"""
        return await self.ny_reversal_strategy(data, symbol, timeframe)
    
    async def optimal_trade_entry_refined_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Refined Optimal Trade Entry Strategy"""
        return await self.optimal_trade_entry_strategy(data, symbol, timeframe)
    
    # Risk Management & Execution Strategies (simplified implementations)
    async def trade_journaling_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Trade Journaling Strategy - Analyze historical setups for performance"""
        setups = []
        
        # Get various strategy setups for analysis
        ob_setups = await self.order_block_strategy(data, symbol, timeframe)
        fvg_setups = await self.fair_value_gap_strategy(data, symbol, timeframe)
        
        # Add performance metadata for journaling
        all_setups = ob_setups + fvg_setups
        for setup in all_setups:
            # Mark setups with journaling metadata
            setup.metadata = {
                "journal_entry": True,
                "strategy_type": setup.setup_type.value,
                "analysis_timestamp": datetime.utcnow()
            }
            setups.append(setup)
        
        return setups[:5]  # Limit for demonstration
    
    async def entry_models_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Entry Models Strategy - FVG, OB, Breaker entries"""
        fvg_setups = await self.fair_value_gap_strategy(data, symbol, timeframe)
        ob_setups = await self.order_block_strategy(data, symbol, timeframe)
        breaker_setups = await self.breaker_block_strategy(data, symbol, timeframe)
        return fvg_setups + ob_setups + breaker_setups
    
    async def exit_models_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Exit Models Strategy - Advanced exit techniques (partial TP, trailing stops)"""
        base_setups = await self.order_block_strategy(data, symbol, timeframe)
        enhanced_setups = []
        
        for setup in base_setups:
            # Enhance with advanced exit models
            if len(setup.take_profit) < 3:
                # Add partial take profit levels
                entry_price = setup.entry_price
                tp_distance = abs(setup.take_profit[0] - entry_price)
                
                enhanced_tp = [
                    entry_price + (tp_distance * 0.5),  # 50% TP
                    entry_price + (tp_distance * 1.0),  # 100% TP  
                    entry_price + (tp_distance * 1.618) # 161.8% TP (Fibonacci extension)
                ]
                
                if setup.direction == TradeDirection.SHORT:
                    enhanced_tp = [entry_price - d for d in [tp_distance * 0.5, tp_distance * 1.0, tp_distance * 1.618]]
                
                setup.take_profit = enhanced_tp
                setup.metadata = {"exit_model": "partial_tp", "tp_levels": 3}
                
            enhanced_setups.append(setup)
        
        return enhanced_setups
    
    async def risk_reward_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Risk-Reward Optimization Strategy - Filter setups by minimum RRR"""
        all_setups = []
        
        # Get setups from various strategies
        ob_setups = await self.order_block_strategy(data, symbol, timeframe)
        fvg_setups = await self.fair_value_gap_strategy(data, symbol, timeframe)
        all_setups.extend(ob_setups + fvg_setups)
        
        # Filter by minimum risk-reward ratio
        MIN_RRR = 2.0  # Minimum 1:2 risk-reward
        high_rrr_setups = []
        
        for setup in all_setups:
            if setup.risk_reward_ratio >= MIN_RRR:
                setup.metadata = {"rrr_filter": True, "min_rrr": MIN_RRR}
                high_rrr_setups.append(setup)
        
        # Sort by RRR descending
        high_rrr_setups.sort(key=lambda x: x.risk_reward_ratio, reverse=True)
        
        return high_rrr_setups[:3]  # Top 3 highest RRR setups
    
    async def position_sizing_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Position Sizing Strategy - Calculate optimal position sizes based on risk"""
        base_setups = await self.order_block_strategy(data, symbol, timeframe)
        
        # Portfolio parameters
        ACCOUNT_SIZE = 10000  # $10,000 account
        RISK_PER_TRADE = 0.02  # 2% risk per trade
        MAX_RISK_AMOUNT = ACCOUNT_SIZE * RISK_PER_TRADE
        
        sized_setups = []
        for setup in base_setups:
            # Calculate position size based on stop loss distance
            risk_per_unit = abs(setup.entry_price - setup.stop_loss)
            
            if risk_per_unit > 0:
                position_size = MAX_RISK_AMOUNT / risk_per_unit
                
                # Add position sizing metadata
                setup.metadata = {
                    "position_size": round(position_size, 2),
                    "risk_amount": MAX_RISK_AMOUNT,
                    "risk_per_unit": risk_per_unit,
                    "account_risk_pct": RISK_PER_TRADE * 100
                }
                sized_setups.append(setup)
        
        return sized_setups
    
    async def drawdown_control_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Drawdown Control Strategy - Manage maximum drawdown exposure"""
        base_setups = await self.order_block_strategy(data, symbol, timeframe)
        
        # Simulate current portfolio state
        current_drawdown = 0.05  # 5% current drawdown
        MAX_DRAWDOWN = 0.15  # 15% maximum allowed drawdown
        
        controlled_setups = []
        
        # Only allow new setups if we're within drawdown limits
        if current_drawdown < MAX_DRAWDOWN:
            # Reduce position sizes during drawdown periods
            drawdown_factor = 1 - (current_drawdown / MAX_DRAWDOWN)
            
            for setup in base_setups:
                setup.metadata = {
                    "drawdown_control": True,
                    "current_drawdown": current_drawdown,
                    "max_drawdown": MAX_DRAWDOWN,
                    "position_reduction": round((1 - drawdown_factor) * 100, 1)
                }
                setup.confidence *= drawdown_factor  # Reduce confidence during drawdown
                controlled_setups.append(setup)
        
        return controlled_setups[:2]  # Limit trades during drawdown
    
    async def compounding_models_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Compounding Models Strategy - Dynamic position sizing based on account growth"""
        base_setups = await self.order_block_strategy(data, symbol, timeframe)
        
        # Simulate account growth
        initial_account = 10000
        current_account = 12500  # 25% growth
        growth_factor = current_account / initial_account
        
        compounded_setups = []
        
        for setup in base_setups:
            # Increase position size based on account growth
            base_risk = 0.02  # 2% base risk
            compounded_risk = min(base_risk * growth_factor, 0.05)  # Cap at 5%
            
            setup.metadata = {
                "compounding": True,
                "initial_account": initial_account,
                "current_account": current_account,
                "growth_factor": round(growth_factor, 2),
                "base_risk": base_risk,
                "compounded_risk": round(compounded_risk, 3)
            }
            
            # Adjust confidence based on compounding success
            setup.confidence *= min(growth_factor, 1.2)  # Bonus confidence, capped
            compounded_setups.append(setup)
        
        return compounded_setups
    
    async def daily_loss_limits_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Daily Loss Limits Strategy - Enforce daily loss limits"""
        
        # Simulate daily PnL tracking
        daily_loss = -150  # $150 loss today
        DAILY_LOSS_LIMIT = -200  # $200 daily loss limit
        
        # Only generate setups if we haven't hit daily limit
        if daily_loss > DAILY_LOSS_LIMIT:
            remaining_risk = abs(DAILY_LOSS_LIMIT - daily_loss)
            
            # Get conservative setups only
            setups = await self.risk_reward_strategy(data, symbol, timeframe)
            
            # Filter to low-risk setups only
            safe_setups = []
            for setup in setups:
                if setup.risk_reward_ratio >= 3.0:  # Higher RRR requirement
                    setup.metadata = {
                        "daily_limit_control": True,
                        "daily_loss": daily_loss,
                        "loss_limit": DAILY_LOSS_LIMIT,
                        "remaining_risk": remaining_risk,
                        "conservative_mode": True
                    }
                    safe_setups.append(setup)
            
            return safe_setups[:1]  # Only one setup near daily limit
        
        # Return empty if daily limit hit
        return []
    
    async def probability_profiles_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Probability Profiles Strategy - A+, B, C setups"""
        # Classify setups by probability
        all_setups = []
        
        # Get various setup types
        ob_setups = await self.order_block_strategy(data, symbol, timeframe)
        fvg_setups = await self.fair_value_gap_strategy(data, symbol, timeframe)
        
        # Classify by confidence (A+ = high confidence, B = medium, C = low)
        for setup in ob_setups + fvg_setups:
            if setup.confidence >= 0.8:
                setup.setup_type = "A+ Setup"
            elif setup.confidence >= 0.7:
                setup.setup_type = "B Setup"
            else:
                setup.setup_type = "C Setup"
            all_setups.append(setup)
        
        return all_setups
    
    # Advanced Concepts
    async def high_probability_scenarios_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """High Probability Trade Scenarios - HTF bias + LTF confirmation"""
        # This would combine multiple timeframe analysis
        htf_setups = await self.weekly_bias_strategy(data, symbol, timeframe)
        ltf_setups = await self.order_block_strategy(data, symbol, timeframe)
        
        # Filter for confluence
        high_prob_setups = []
        for ltf_setup in ltf_setups:
            for htf_setup in htf_setups:
                if ltf_setup.direction == htf_setup.direction:
                    ltf_setup.confidence = min(ltf_setup.confidence + 0.1, 1.0)
                    high_prob_setups.append(ltf_setup)
                    break
        
        return high_prob_setups
    
    async def liquidity_runs_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Liquidity Runs Strategy"""
        return await self.liquidity_raid_strategy(data, symbol, timeframe)
    
    async def reversals_continuations_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Reversals vs Continuations Strategy"""
        reversal_setups = await self.ny_reversal_strategy(data, symbol, timeframe)
        continuation_setups = await self.market_structure_strategy(data, symbol, timeframe)
        return reversal_setups + continuation_setups
    
    async def accumulation_distribution_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Accumulation & Distribution Schematics Strategy"""
        return await self.market_maker_models_strategy(data, symbol, timeframe)
    
    async def order_flow_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Order Flow Strategy"""
        return await self.market_maker_models_strategy(data, symbol, timeframe)
    
    async def high_low_day_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """High/Low of Day Identification Strategy"""
        return await self.daily_bias_strategy(data, symbol, timeframe)
    
    async def range_expansion_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Range Expansion Strategy"""
        return await self.range_expectations_strategy(data, symbol, timeframe)
    
    async def inside_outside_day_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Inside Day / Outside Day Strategy"""
        setups = []
        
        if len(data) < 3:
            return setups
        
        # Check for inside/outside day patterns
        for i in range(2, len(data)):
            today = data.iloc[i]
            yesterday = data.iloc[i-1]
            
            # Inside day: today's range is within yesterday's range
            if (today['high'] < yesterday['high'] and today['low'] > yesterday['low']):
                # Anticipate breakout
                current_price = today['close']
                
                # Bullish breakout setup
                entry_price = yesterday['high']
                stop_loss = yesterday['low']
                take_profit = [entry_price + (entry_price - stop_loss)]
                
                setup = TradeSetup(
                    symbol=symbol,
                    direction=TradeDirection.LONG,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward_ratio=1.0,
                    setup_type=ICTConcept.INSIDE_DAY,
                    confidence=0.6,
                    timestamp=today['timestamp'],
                    timeframe=timeframe
                )
                setups.append(setup)
            
            # Outside day: today's range engulfs yesterday's range
            elif (today['high'] > yesterday['high'] and today['low'] < yesterday['low']):
                # Continuation setup in direction of close
                direction = TradeDirection.LONG if today['close'] > today['open'] else TradeDirection.SHORT
                
                if direction == TradeDirection.LONG:
                    entry_price = today['close']
                    stop_loss = today['low']
                    take_profit = [entry_price + (today['high'] - today['low'])]
                else:
                    entry_price = today['close']
                    stop_loss = today['high']
                    take_profit = [entry_price - (today['high'] - today['low'])]
                
                setup = TradeSetup(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_reward_ratio=1.0,
                    setup_type=ICTConcept.OUTSIDE_DAY,
                    confidence=0.7,
                    timestamp=today['timestamp'],
                    timeframe=timeframe
                )
                setups.append(setup)
        
        return setups
    
    async def weekly_profiles_advanced_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Advanced Weekly Profiles Strategy"""
        return await self.weekly_profiles_strategy(data, symbol, timeframe)
    
    async def ipda_theory_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """IPDA Theory Strategy"""
        # Interbank Price Delivery Algorithm theory
        setups = []
        
        # Look for algorithmic price delivery patterns
        current_time = datetime.utcnow()
        hour = current_time.hour
        minute = current_time.minute
        
        # Key algorithmic times (simplified)
        algo_times = [
            (8, 30),   # London open
            (9, 0),    # Frankfurt open
            (13, 30),  # NY open
            (14, 0),   # NY cash open
            (20, 0),   # Sydney open
        ]
        
        # Check if we're near an algorithmic time
        for algo_hour, algo_minute in algo_times:
            time_diff = abs((hour * 60 + minute) - (algo_hour * 60 + algo_minute))
            
            if time_diff <= 15:  # Within 15 minutes
                # Look for algorithmic price movements
                if len(data) >= 5:
                    recent_data = data.iloc[-5:]
                    price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
                    
                    if abs(price_change) > 0.002:  # 0.2% move
                        current_price = data.iloc[-1]['close']
                        direction = TradeDirection.LONG if price_change > 0 else TradeDirection.SHORT
                        
                        if direction == TradeDirection.LONG:
                            entry_price = current_price
                            stop_loss = current_price * 0.998
                            take_profit = [current_price * 1.005]
                        else:
                            entry_price = current_price
                            stop_loss = current_price * 1.002
                            take_profit = [current_price * 0.995]
                        
                        setup = TradeSetup(
                            symbol=symbol,
                            direction=direction,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            risk_reward_ratio=2.5,
                            setup_type=ICTConcept.IPDA_THEORY,
                            confidence=0.7,
                            timestamp=datetime.utcnow(),
                            timeframe=timeframe
                        )
                        setups.append(setup)
                break
        
        return setups
    
    async def algo_price_delivery_strategy(self, data: pd.DataFrame, symbol: str, timeframe: TimeFrame) -> List[TradeSetup]:
        """Algorithmic Price Delivery Strategy"""
        return await self.ipda_theory_strategy(data, symbol, timeframe)