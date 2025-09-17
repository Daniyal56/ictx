import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
# import cv2  # Temporarily disabled for compatibility

from app.models import TimeFrame, TradeDirection

class PatternRecognizer:
    """Advanced pattern recognition for ICT concepts using machine learning"""
    
    def __init__(self):
        self.pattern_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Pattern templates for ALL 65 ICT concepts
        self.pattern_templates = {
            # Core ICT Concepts (1-20)
            'market_structure': self._get_market_structure_template(),
            'liquidity': self._get_liquidity_template(),
            'liquidity_pools': self._get_liquidity_pools_template(),
            'order_block': self._get_order_block_template(),
            'breaker_block': self._get_breaker_template(),
            'fair_value_gap': self._get_fvg_template(),
            'rejection_block': self._get_rejection_template(),
            'mitigation_block': self._get_mitigation_template(),
            'supply_demand_zones': self._get_supply_demand_template(),
            'premium_discount': self._get_premium_discount_template(),
            'dealing_ranges': self._get_dealing_ranges_template(),
            'swing_points': self._get_swing_points_template(),
            'market_maker_models': self._get_market_maker_template(),
            'judas_swing': self._get_judas_swing_template(),
            'turtle_soup': self._get_turtle_soup_template(),
            'power_of_three': self._get_power_of_three_template(),
            'optimal_trade_entry': self._get_ote_template(),
            'smt_divergence': self._get_smt_divergence_template(),
            'liquidity_voids': self._get_liquidity_voids_template(),
            
            # Time & Price Theory (21-30)
            'killzones': self._get_killzones_template(),
            'session_opens': self._get_session_opens_template(),
            'fibonacci_ratios': self._get_fibonacci_template(),
            'range_expectations': self._get_range_expectations_template(),
            'session_liquidity_raids': self._get_session_raids_template(),
            'weekly_profiles': self._get_weekly_profiles_template(),
            'daily_bias': self._get_daily_bias_template(),
            'weekly_bias': self._get_weekly_bias_template(),
            'monthly_bias': self._get_monthly_bias_template(),
            'time_of_day': self._get_time_of_day_template(),
            
            # Advanced Concepts (40-50)
            'high_probability_scenarios': self._get_high_prob_template(),
            'liquidity_runs': self._get_liquidity_runs_template(),
            'reversals_continuations': self._get_reversals_template(),
            'accumulation_distribution': self._get_amd_template(),
            'order_flow': self._get_order_flow_template(),
            'high_low_day': self._get_high_low_day_template(),
            'range_expansion': self._get_range_expansion_template(),
            'inside_outside_day': self._get_inside_outside_template(),
            'weekly_profiles_advanced': self._get_weekly_advanced_template(),
            'ipda_theory': self._get_ipda_template(),
            'algo_price_delivery': self._get_algo_delivery_template(),
            
            # Strategy Playbooks (51-65)
            'silver_bullet': self._get_silver_bullet_template(),
            'asian_range': self._get_asian_range_template(),
            'ny_reversal': self._get_ny_reversal_template(),
            'london_killzone': self._get_london_killzone_template(),
            'fvg_sniper': self._get_fvg_sniper_template(),
            'order_block_refined': self._get_ob_refined_template(),
            'breaker_block_refined': self._get_breaker_refined_template(),
            'rejection_block_refined': self._get_rejection_refined_template(),
            'smt_divergence_refined': self._get_smt_refined_template(),
            'turtle_soup_refined': self._get_turtle_soup_refined_template(),
            'power_of_three_refined': self._get_power_three_refined_template(),
            'daily_bias_liquidity_raid': self._get_daily_bias_raid_template(),
            'am_session_bias': self._get_am_session_template(),
            'pm_session_reversal': self._get_pm_session_template(),
            'optimal_trade_entry_refined': self._get_ote_refined_template()
        }
    
    async def detect_patterns(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        lookback_periods: int = 100
    ) -> List[Dict[str, Any]]:
        """Detect ALL ICT patterns in market data using comprehensive analysis"""
        
        # Get market data
        data = await self._get_market_data(symbol, timeframe, lookback_periods)
        
        detected_patterns = []
        
        # Detect ALL pattern types from the 65 ICT concepts
        for pattern_name, template in self.pattern_templates.items():
            try:
                patterns = await self._detect_pattern_type(data, pattern_name, template)
                detected_patterns.extend(patterns)
            except Exception as e:
                print(f"Error detecting {pattern_name}: {str(e)}")
                continue
        
        # Sort by confidence and recency
        detected_patterns.sort(key=lambda x: (x['confidence'], x['timestamp']), reverse=True)
        
        return detected_patterns[:50]  # Return top 50 patterns for comprehensive analysis
    
    async def _detect_pattern_type(
        self, 
        data: pd.DataFrame, 
        pattern_name: str, 
        template: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect specific pattern type in data - Enhanced for ALL 65 ICT concepts"""
        
        patterns = []
        
        # Core ICT Concepts (1-20)
        if pattern_name == 'market_structure':
            patterns = self._detect_market_structure(data)
        elif pattern_name == 'liquidity':
            patterns = self._detect_liquidity_patterns(data)
        elif pattern_name == 'liquidity_pools':
            patterns = self._detect_liquidity_pools(data)
        elif pattern_name == 'order_block':
            patterns = self._detect_order_blocks(data)
        elif pattern_name == 'breaker_block':
            patterns = self._detect_breaker_blocks(data)
        elif pattern_name == 'fair_value_gap':
            patterns = self._detect_fair_value_gaps(data)
        elif pattern_name == 'rejection_block':
            patterns = self._detect_rejection_blocks(data)
        elif pattern_name == 'mitigation_block':
            patterns = self._detect_mitigation_blocks(data)
        elif pattern_name == 'supply_demand_zones':
            patterns = self._detect_supply_demand_zones(data)
        elif pattern_name == 'premium_discount':
            patterns = self._detect_premium_discount(data)
        elif pattern_name == 'dealing_ranges':
            patterns = self._detect_dealing_ranges(data)
        elif pattern_name == 'swing_points':
            patterns = self._detect_swing_points(data)
        elif pattern_name == 'market_maker_models':
            patterns = self._detect_market_maker_models(data)
        elif pattern_name == 'judas_swing':
            patterns = self._detect_judas_swing(data)
        elif pattern_name == 'turtle_soup':
            patterns = self._detect_turtle_soup(data)
        elif pattern_name == 'power_of_three':
            patterns = self._detect_power_of_three(data)
        elif pattern_name == 'optimal_trade_entry':
            patterns = self._detect_optimal_trade_entry(data)
        elif pattern_name == 'smt_divergence':
            patterns = self._detect_smt_divergence(data)
        elif pattern_name == 'liquidity_voids':
            patterns = self._detect_liquidity_voids(data)
        
        # Time & Price Theory (21-30)
        elif pattern_name == 'killzones':
            patterns = self._detect_killzone_patterns(data)
        elif pattern_name == 'session_opens':
            patterns = self._detect_session_open_patterns(data)
        elif pattern_name == 'fibonacci_ratios':
            patterns = self._detect_fibonacci_patterns(data)
        elif pattern_name == 'silver_bullet':
            patterns = self._detect_silver_bullet_patterns(data)
        elif pattern_name == 'asian_range':
            patterns = self._detect_asian_range_patterns(data)
        elif pattern_name == 'ny_reversal':
            patterns = self._detect_ny_reversal_patterns(data)
        elif pattern_name == 'london_killzone':
            patterns = self._detect_london_killzone_patterns(data)
        
        # Advanced Concepts (40-50)
        elif pattern_name == 'liquidity_runs':
            patterns = self._detect_liquidity_run_patterns(data)
        elif pattern_name == 'accumulation_distribution':
            patterns = self._detect_accumulation_distribution(data)
        elif pattern_name == 'order_flow':
            patterns = self._detect_order_flow_patterns(data)
        elif pattern_name == 'ipda_theory':
            patterns = self._detect_ipda_patterns(data)
        elif pattern_name == 'algo_price_delivery':
            patterns = self._detect_algo_delivery_patterns(data)
        
        # Default fallback for any missing patterns
        else:
            patterns = self._detect_generic_pattern(data, pattern_name)
        
        # Add pattern name and template match to each detection
        for pattern in patterns:
            pattern['pattern_type'] = pattern_name
            pattern['template_match'] = self._calculate_template_match(pattern, template)
        
        return patterns
    
    def _detect_order_blocks(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect order block patterns"""
        patterns = []
        
        for i in range(20, len(data) - 5):
            # Look for strong impulse candle followed by consolidation
            current_candle = data.iloc[i]
            body_size = abs(current_candle['close'] - current_candle['open'])
            candle_range = current_candle['high'] - current_candle['low']
            
            # Strong bullish candle criteria
            if (current_candle['close'] > current_candle['open'] and 
                body_size > candle_range * 0.7):
                
                # Check for preceding consolidation
                prev_range = np.mean([
                    data.iloc[j]['high'] - data.iloc[j]['low'] 
                    for j in range(i-10, i)
                ])
                
                if candle_range > prev_range * 2:  # Significant breakout
                    # Check if price respects this level later
                    respect_count = 0
                    for j in range(i+1, min(i+20, len(data))):
                        if (data.iloc[j]['low'] <= current_candle['high'] and 
                            data.iloc[j]['close'] > current_candle['low']):
                            respect_count += 1
                    
                    confidence = min(respect_count / 10, 1.0)
                    
                    if confidence > 0.3:
                        patterns.append({
                            'timestamp': current_candle['timestamp'],
                            'high': current_candle['high'],
                            'low': current_candle['low'],
                            'direction': 'bullish',
                            'confidence': confidence,
                            'strength': body_size / prev_range,
                            'respect_count': respect_count,
                            'status': 'active' if respect_count > 0 else 'untested'
                        })
            
            # Strong bearish candle criteria
            elif (current_candle['close'] < current_candle['open'] and 
                  body_size > candle_range * 0.7):
                
                prev_range = np.mean([
                    data.iloc[j]['high'] - data.iloc[j]['low'] 
                    for j in range(i-10, i)
                ])
                
                if candle_range > prev_range * 2:
                    respect_count = 0
                    for j in range(i+1, min(i+20, len(data))):
                        if (data.iloc[j]['high'] >= current_candle['low'] and 
                            data.iloc[j]['close'] < current_candle['high']):
                            respect_count += 1
                    
                    confidence = min(respect_count / 10, 1.0)
                    
                    if confidence > 0.3:
                        patterns.append({
                            'timestamp': current_candle['timestamp'],
                            'high': current_candle['high'],
                            'low': current_candle['low'],
                            'direction': 'bearish',
                            'confidence': confidence,
                            'strength': body_size / prev_range,
                            'respect_count': respect_count,
                            'status': 'active' if respect_count > 0 else 'untested'
                        })
        
        return patterns
    
    def _detect_fair_value_gaps(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Fair Value Gap patterns"""
        patterns = []
        
        for i in range(2, len(data)):
            # Bullish FVG: Gap between candle[i-2].high and candle[i].low
            if data.iloc[i-2]['high'] < data.iloc[i]['low']:
                gap_size = data.iloc[i]['low'] - data.iloc[i-2]['high']
                
                # Check if gap gets filled later
                filled = False
                fill_percentage = 0
                
                for j in range(i+1, min(i+50, len(data))):
                    if data.iloc[j]['low'] <= data.iloc[i-2]['high']:
                        filled = True
                        break
                    elif data.iloc[j]['low'] < data.iloc[i]['low']:
                        fill_percentage = (data.iloc[i]['low'] - data.iloc[j]['low']) / gap_size
                
                patterns.append({
                    'timestamp': data.iloc[i-1]['timestamp'],
                    'top': data.iloc[i]['low'],
                    'bottom': data.iloc[i-2]['high'],
                    'direction': 'bullish',
                    'size': gap_size,
                    'confidence': 0.8 if not filled else 0.6,
                    'filled': filled,
                    'fill_percentage': fill_percentage,
                    'status': 'filled' if filled else 'open'
                })
            
            # Bearish FVG: Gap between candle[i-2].low and candle[i].high
            elif data.iloc[i-2]['low'] > data.iloc[i]['high']:
                gap_size = data.iloc[i-2]['low'] - data.iloc[i]['high']
                
                filled = False
                fill_percentage = 0
                
                for j in range(i+1, min(i+50, len(data))):
                    if data.iloc[j]['high'] >= data.iloc[i-2]['low']:
                        filled = True
                        break
                    elif data.iloc[j]['high'] > data.iloc[i]['high']:
                        fill_percentage = (data.iloc[j]['high'] - data.iloc[i]['high']) / gap_size
                
                patterns.append({
                    'timestamp': data.iloc[i-1]['timestamp'],
                    'top': data.iloc[i-2]['low'],
                    'bottom': data.iloc[i]['high'],
                    'direction': 'bearish',
                    'size': gap_size,
                    'confidence': 0.8 if not filled else 0.6,
                    'filled': filled,
                    'fill_percentage': fill_percentage,
                    'status': 'filled' if filled else 'open'
                })
        
        return patterns
    
    def _detect_breaker_blocks(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Breaker Block patterns (failed order blocks)"""
        patterns = []
        
        # First find order blocks
        order_blocks = self._detect_order_blocks(data)
        
        for ob in order_blocks:
            ob_timestamp = ob['timestamp']
            ob_index = data[data['timestamp'] == ob_timestamp].index[0]
            
            # Look for break of order block
            if ob['direction'] == 'bullish':
                # Look for break below order block low
                for j in range(ob_index + 1, min(ob_index + 30, len(data))):
                    if data.iloc[j]['close'] < ob['low']:
                        # Order block broken, now it becomes a breaker block (resistance)
                        
                        # Check if it acts as resistance later
                        resistance_count = 0
                        for k in range(j + 1, min(j + 20, len(data))):
                            if (data.iloc[k]['high'] >= ob['low'] and 
                                data.iloc[k]['close'] < ob['high']):
                                resistance_count += 1
                        
                        if resistance_count > 0:
                            patterns.append({
                                'timestamp': data.iloc[j]['timestamp'],
                                'original_ob_timestamp': ob_timestamp,
                                'high': ob['high'],
                                'low': ob['low'],
                                'direction': 'bearish',  # Now acts as resistance
                                'confidence': min(resistance_count / 10, 0.9),
                                'break_timestamp': data.iloc[j]['timestamp'],
                                'resistance_count': resistance_count,
                                'status': 'active'
                            })
                        break
            
            else:  # Bearish order block
                # Look for break above order block high
                for j in range(ob_index + 1, min(ob_index + 30, len(data))):
                    if data.iloc[j]['close'] > ob['high']:
                        # Order block broken, now it becomes a breaker block (support)
                        
                        support_count = 0
                        for k in range(j + 1, min(j + 20, len(data))):
                            if (data.iloc[k]['low'] <= ob['high'] and 
                                data.iloc[k]['close'] > ob['low']):
                                support_count += 1
                        
                        if support_count > 0:
                            patterns.append({
                                'timestamp': data.iloc[j]['timestamp'],
                                'original_ob_timestamp': ob_timestamp,
                                'high': ob['high'],
                                'low': ob['low'],
                                'direction': 'bullish',  # Now acts as support
                                'confidence': min(support_count / 10, 0.9),
                                'break_timestamp': data.iloc[j]['timestamp'],
                                'support_count': support_count,
                                'status': 'active'
                            })
                        break
        
        return patterns
    
    def _detect_liquidity_pools(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Liquidity Pool patterns (equal highs/lows)"""
        patterns = []
        tolerance = 0.001  # 0.1% tolerance for "equal" levels
        
        # Detect equal highs
        highs = data['high'].values
        for i in range(10, len(highs) - 10):
            current_high = highs[i]
            
            # Look for other highs at similar level
            equal_highs = []
            for j in range(max(0, i-20), min(len(highs), i+20)):
                if (j != i and 
                    abs(highs[j] - current_high) / current_high < tolerance):
                    equal_highs.append(j)
            
            if len(equal_highs) >= 1:  # At least 2 equal highs (including current)
                # Check if this level gets swept
                swept = False
                sweep_timestamp = None
                
                for j in range(i + 1, min(i + 30, len(data))):
                    if data.iloc[j]['high'] > current_high * (1 + tolerance):
                        swept = True
                        sweep_timestamp = data.iloc[j]['timestamp']
                        break
                
                patterns.append({
                    'timestamp': data.iloc[i]['timestamp'],
                    'price': current_high,
                    'type': 'equal_highs',
                    'equal_count': len(equal_highs) + 1,
                    'confidence': min(len(equal_highs) / 3, 1.0),
                    'swept': swept,
                    'sweep_timestamp': sweep_timestamp,
                    'strength': len(equal_highs) / 5,
                    'status': 'swept' if swept else 'active'
                })
        
        # Detect equal lows
        lows = data['low'].values
        for i in range(10, len(lows) - 10):
            current_low = lows[i]
            
            equal_lows = []
            for j in range(max(0, i-20), min(len(lows), i+20)):
                if (j != i and 
                    abs(lows[j] - current_low) / current_low < tolerance):
                    equal_lows.append(j)
            
            if len(equal_lows) >= 1:
                swept = False
                sweep_timestamp = None
                
                for j in range(i + 1, min(i + 30, len(data))):
                    if data.iloc[j]['low'] < current_low * (1 - tolerance):
                        swept = True
                        sweep_timestamp = data.iloc[j]['timestamp']
                        break
                
                patterns.append({
                    'timestamp': data.iloc[i]['timestamp'],
                    'price': current_low,
                    'type': 'equal_lows',
                    'equal_count': len(equal_lows) + 1,
                    'confidence': min(len(equal_lows) / 3, 1.0),
                    'swept': swept,
                    'sweep_timestamp': sweep_timestamp,
                    'strength': len(equal_lows) / 5,
                    'status': 'swept' if swept else 'active'
                })
        
        return patterns
    
    def _detect_rejection_blocks(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Rejection Block patterns"""
        patterns = []
        
        for i in range(10, len(data) - 5):
            current_candle = data.iloc[i]
            
            # Look for candles with long wicks (rejection)
            total_range = current_candle['high'] - current_candle['low']
            body_size = abs(current_candle['close'] - current_candle['open'])
            
            if total_range > 0:
                upper_wick = current_candle['high'] - max(current_candle['open'], current_candle['close'])
                lower_wick = min(current_candle['open'], current_candle['close']) - current_candle['low']
                
                # Bullish rejection (long lower wick)
                if (lower_wick > total_range * 0.6 and 
                    current_candle['close'] > current_candle['open']):
                    
                    # Check if this level holds as support
                    support_count = 0
                    for j in range(i + 1, min(i + 15, len(data))):
                        if (data.iloc[j]['low'] >= current_candle['low'] and 
                            data.iloc[j]['low'] <= current_candle['low'] + total_range * 0.2):
                            support_count += 1
                    
                    if support_count > 0:
                        patterns.append({
                            'timestamp': current_candle['timestamp'],
                            'high': current_candle['high'],
                            'low': current_candle['low'],
                            'rejection_level': current_candle['low'],
                            'direction': 'bullish',
                            'wick_ratio': lower_wick / total_range,
                            'confidence': min(support_count / 10, 0.9),
                            'support_count': support_count,
                            'status': 'active'
                        })
                
                # Bearish rejection (long upper wick)
                elif (upper_wick > total_range * 0.6 and 
                      current_candle['close'] < current_candle['open']):
                    
                    resistance_count = 0
                    for j in range(i + 1, min(i + 15, len(data))):
                        if (data.iloc[j]['high'] <= current_candle['high'] and 
                            data.iloc[j]['high'] >= current_candle['high'] - total_range * 0.2):
                            resistance_count += 1
                    
                    if resistance_count > 0:
                        patterns.append({
                            'timestamp': current_candle['timestamp'],
                            'high': current_candle['high'],
                            'low': current_candle['low'],
                            'rejection_level': current_candle['high'],
                            'direction': 'bearish',
                            'wick_ratio': upper_wick / total_range,
                            'confidence': min(resistance_count / 10, 0.9),
                            'resistance_count': resistance_count,
                            'status': 'active'
                        })
        
        return patterns
    
    def _detect_mitigation_blocks(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Mitigation Block patterns"""
        patterns = []
        
        # Find order blocks first
        order_blocks = self._detect_order_blocks(data)
        
        for ob in order_blocks:
            ob_timestamp = ob['timestamp']
            ob_index = data[data['timestamp'] == ob_timestamp].index[0]
            
            # Look for partial mitigation (price returns to order block)
            if ob['direction'] == 'bullish':
                for j in range(ob_index + 5, min(ob_index + 50, len(data))):
                    # Check if price returns to order block range
                    if (data.iloc[j]['low'] <= ob['high'] and 
                        data.iloc[j]['high'] >= ob['low']):
                        
                        # Check how much of the order block is mitigated
                        mitigation_percentage = 0
                        if data.iloc[j]['low'] <= ob['low']:
                            mitigation_percentage = 1.0  # Full mitigation
                        else:
                            mitigation_percentage = (ob['high'] - data.iloc[j]['low']) / (ob['high'] - ob['low'])
                        
                        if mitigation_percentage > 0.3:  # At least 30% mitigated
                            patterns.append({
                                'timestamp': data.iloc[j]['timestamp'],
                                'original_ob_timestamp': ob_timestamp,
                                'high': ob['high'],
                                'low': ob['low'],
                                'mitigation_level': data.iloc[j]['low'],
                                'mitigation_percentage': mitigation_percentage,
                                'direction': ob['direction'],
                                'confidence': mitigation_percentage,
                                'status': 'full' if mitigation_percentage >= 0.9 else 'partial'
                            })
                        break
            
            else:  # Bearish order block
                for j in range(ob_index + 5, min(ob_index + 50, len(data))):
                    if (data.iloc[j]['high'] >= ob['low'] and 
                        data.iloc[j]['low'] <= ob['high']):
                        
                        mitigation_percentage = 0
                        if data.iloc[j]['high'] >= ob['high']:
                            mitigation_percentage = 1.0
                        else:
                            mitigation_percentage = (data.iloc[j]['high'] - ob['low']) / (ob['high'] - ob['low'])
                        
                        if mitigation_percentage > 0.3:
                            patterns.append({
                                'timestamp': data.iloc[j]['timestamp'],
                                'original_ob_timestamp': ob_timestamp,
                                'high': ob['high'],
                                'low': ob['low'],
                                'mitigation_level': data.iloc[j]['high'],
                                'mitigation_percentage': mitigation_percentage,
                                'direction': ob['direction'],
                                'confidence': mitigation_percentage,
                                'status': 'full' if mitigation_percentage >= 0.9 else 'partial'
                            })
                        break
        
        return patterns
    
    def _detect_turtle_soup(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Turtle Soup patterns (stop hunt reversals)"""
        patterns = []
        
        # Look for false breakouts
        for i in range(20, len(data) - 10):
            # Find recent high/low levels
            recent_high = max(data['high'].iloc[i-20:i])
            recent_low = min(data['low'].iloc[i-20:i])
            
            current_candle = data.iloc[i]
            
            # False breakout above recent high
            if current_candle['high'] > recent_high:
                # Check if price quickly reverses back below the level
                reversal_found = False
                for j in range(i + 1, min(i + 5, len(data))):
                    if data.iloc[j]['close'] < recent_high:
                        reversal_found = True
                        
                        # Measure the strength of reversal
                        reversal_distance = recent_high - data.iloc[j]['close']
                        breakout_distance = current_candle['high'] - recent_high
                        
                        if reversal_distance > breakout_distance * 0.5:  # Strong reversal
                            patterns.append({
                                'timestamp': current_candle['timestamp'],
                                'breakout_level': recent_high,
                                'false_high': current_candle['high'],
                                'reversal_low': data.iloc[j]['close'],
                                'direction': 'bearish',
                                'strength': reversal_distance / breakout_distance,
                                'confidence': 0.8,
                                'reversal_speed': j - i,  # How quickly it reversed
                                'status': 'confirmed'
                            })
                        break
                
                if not reversal_found:
                    # Potential turtle soup still developing
                    patterns.append({
                        'timestamp': current_candle['timestamp'],
                        'breakout_level': recent_high,
                        'false_high': current_candle['high'],
                        'direction': 'bearish',
                        'confidence': 0.5,
                        'status': 'developing'
                    })
            
            # False breakout below recent low
            elif current_candle['low'] < recent_low:
                reversal_found = False
                for j in range(i + 1, min(i + 5, len(data))):
                    if data.iloc[j]['close'] > recent_low:
                        reversal_found = True
                        
                        reversal_distance = data.iloc[j]['close'] - recent_low
                        breakout_distance = recent_low - current_candle['low']
                        
                        if reversal_distance > breakout_distance * 0.5:
                            patterns.append({
                                'timestamp': current_candle['timestamp'],
                                'breakout_level': recent_low,
                                'false_low': current_candle['low'],
                                'reversal_high': data.iloc[j]['close'],
                                'direction': 'bullish',
                                'strength': reversal_distance / breakout_distance,
                                'confidence': 0.8,
                                'reversal_speed': j - i,
                                'status': 'confirmed'
                            })
                        break
                
                if not reversal_found:
                    patterns.append({
                        'timestamp': current_candle['timestamp'],
                        'breakout_level': recent_low,
                        'false_low': current_candle['low'],
                        'direction': 'bullish',
                        'confidence': 0.5,
                        'status': 'developing'
                    })
        
        return patterns
    
    def _detect_judas_swing(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Judas Swing patterns (false moves at session opens)"""
        patterns = []
        
        # Look for moves at session open times
        for i in range(5, len(data) - 10):
            timestamp = data.iloc[i]['timestamp']
            hour = timestamp.hour
            
            # Check if this is near session open (simplified)
            is_session_open = (hour in [0, 8, 13])  # Midnight, London, NY opens
            
            if is_session_open:
                # Look for initial move followed by reversal
                open_price = data.iloc[i]['open']
                
                # Check for initial directional move
                initial_move_high = max(data['high'].iloc[i:i+3])
                initial_move_low = min(data['low'].iloc[i:i+3])
                
                # Upward judas swing
                if initial_move_high > open_price * 1.002:  # 0.2% move up
                    # Look for reversal below open
                    for j in range(i + 3, min(i + 15, len(data))):
                        if data.iloc[j]['low'] < open_price * 0.998:  # Reversal below open
                            reversal_distance = open_price - data.iloc[j]['low']
                            initial_distance = initial_move_high - open_price
                            
                            patterns.append({
                                'timestamp': timestamp,
                                'session_hour': hour,
                                'open_price': open_price,
                                'false_high': initial_move_high,
                                'reversal_low': data.iloc[j]['low'],
                                'direction': 'bearish',
                                'strength': reversal_distance / initial_distance,
                                'confidence': 0.75,
                                'reversal_time': j - i,
                                'status': 'confirmed'
                            })
                            break
                
                # Downward judas swing
                elif initial_move_low < open_price * 0.998:  # 0.2% move down
                    # Look for reversal above open
                    for j in range(i + 3, min(i + 15, len(data))):
                        if data.iloc[j]['high'] > open_price * 1.002:  # Reversal above open
                            reversal_distance = data.iloc[j]['high'] - open_price
                            initial_distance = open_price - initial_move_low
                            
                            patterns.append({
                                'timestamp': timestamp,
                                'session_hour': hour,
                                'open_price': open_price,
                                'false_low': initial_move_low,
                                'reversal_high': data.iloc[j]['high'],
                                'direction': 'bullish',
                                'strength': reversal_distance / initial_distance,
                                'confidence': 0.75,
                                'reversal_time': j - i,
                                'status': 'confirmed'
                            })
                            break
        
        return patterns
    
    def _detect_power_of_three(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Power of 3 patterns (Accumulation-Manipulation-Distribution)"""
        patterns = []
        
        # Look for 3-phase patterns over longer timeframes
        for i in range(30, len(data) - 30):
            
            # Define phases
            accumulation_data = data.iloc[i-30:i-20]
            manipulation_data = data.iloc[i-20:i-5]
            distribution_data = data.iloc[i-5:i+10] if i+10 < len(data) else data.iloc[i-5:]
            
            # Accumulation phase: consolidation/range
            acc_high = accumulation_data['high'].max()
            acc_low = accumulation_data['low'].min()
            acc_range = acc_high - acc_low
            acc_volatility = accumulation_data['close'].std()
            
            # Manipulation phase: breakout/false move
            manip_high = manipulation_data['high'].max()
            manip_low = manipulation_data['low'].min()
            
            # Distribution phase: reversal and trend
            if len(distribution_data) > 5:
                dist_trend = (distribution_data['close'].iloc[-1] - distribution_data['close'].iloc[0]) / distribution_data['close'].iloc[0]
                
                # Bullish Power of 3
                if (manip_low < acc_low and  # Manipulation sweeps lows
                    manip_high > acc_high and  # Then breaks highs
                    dist_trend > 0.01):  # Distribution phase shows uptrend
                    
                    patterns.append({
                        'timestamp': data.iloc[i]['timestamp'],
                        'accumulation_range': [acc_low, acc_high],
                        'manipulation_low': manip_low,
                        'manipulation_high': manip_high,
                        'distribution_trend': dist_trend,
                        'direction': 'bullish',
                        'confidence': min(abs(dist_trend) * 10, 0.9),
                        'phase': 'distribution',
                        'accumulation_volatility': acc_volatility,
                        'status': 'active'
                    })
                
                # Bearish Power of 3
                elif (manip_high > acc_high and  # Manipulation sweeps highs
                      manip_low < acc_low and  # Then breaks lows
                      dist_trend < -0.01):  # Distribution phase shows downtrend
                    
                    patterns.append({
                        'timestamp': data.iloc[i]['timestamp'],
                        'accumulation_range': [acc_low, acc_high],
                        'manipulation_low': manip_low,
                        'manipulation_high': manip_high,
                        'distribution_trend': dist_trend,
                        'direction': 'bearish',
                        'confidence': min(abs(dist_trend) * 10, 0.9),
                        'phase': 'distribution',
                        'accumulation_volatility': acc_volatility,
                        'status': 'active'
                    })
        
        return patterns
    
    def _detect_smt_divergence(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect SMT Divergence patterns"""
        patterns = []
        
        # This is a simplified version - real SMT divergence requires correlated pairs
        # For demo purposes, we'll look for internal divergence patterns
        
        # Calculate a simple momentum indicator
        momentum = []
        for i in range(10, len(data)):
            current_momentum = (data['close'].iloc[i] - data['close'].iloc[i-10]) / data['close'].iloc[i-10]
            momentum.append(current_momentum)
        
        # Look for divergence between price and momentum
        for i in range(20, len(momentum) - 5):
            current_price = data['close'].iloc[i + 10]  # Adjust for momentum offset
            previous_price = data['close'].iloc[i - 10 + 10]
            
            current_momentum = momentum[i]
            previous_momentum = momentum[i - 10]
            
            # Bullish divergence: price makes lower low, momentum makes higher low
            if (current_price < previous_price and 
                current_momentum > previous_momentum and
                current_momentum < 0 and previous_momentum < 0):
                
                patterns.append({
                    'timestamp': data.iloc[i + 10]['timestamp'],
                    'price_low': current_price,
                    'previous_price_low': previous_price,
                    'momentum_current': current_momentum,
                    'momentum_previous': previous_momentum,
                    'direction': 'bullish',
                    'divergence_strength': abs(current_momentum - previous_momentum),
                    'confidence': 0.7,
                    'type': 'regular_bullish',
                    'status': 'active'
                })
            
            # Bearish divergence: price makes higher high, momentum makes lower high
            elif (current_price > previous_price and 
                  current_momentum < previous_momentum and
                  current_momentum > 0 and previous_momentum > 0):
                
                patterns.append({
                    'timestamp': data.iloc[i + 10]['timestamp'],
                    'price_high': current_price,
                    'previous_price_high': previous_price,
                    'momentum_current': current_momentum,
                    'momentum_previous': previous_momentum,
                    'direction': 'bearish',
                    'divergence_strength': abs(current_momentum - previous_momentum),
                    'confidence': 0.7,
                    'type': 'regular_bearish',
                    'status': 'active'
                })
        
        return patterns
    
    # Pattern template methods
    def _get_order_block_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['strong_candle', 'preceding_consolidation', 'future_respect'],
            'confidence_factors': ['body_ratio', 'range_multiple', 'respect_count'],
            'timeframe_sensitivity': 'medium'
        }
    
    def _get_fvg_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['price_gap', 'gap_size', 'fill_behavior'],
            'confidence_factors': ['gap_size_ratio', 'fill_speed', 'respect_at_edges'],
            'timeframe_sensitivity': 'high'
        }
    
    def _get_breaker_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['broken_order_block', 'role_reversal', 'new_respect'],
            'confidence_factors': ['break_strength', 'new_respect_count', 'hold_duration'],
            'timeframe_sensitivity': 'medium'
        }
    
    def _get_liquidity_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['equal_levels', 'level_count', 'sweep_behavior'],
            'confidence_factors': ['level_precision', 'touch_count', 'sweep_reaction'],
            'timeframe_sensitivity': 'low'
        }
    
    def _get_rejection_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['long_wick', 'wick_ratio', 'level_respect'],
            'confidence_factors': ['wick_size', 'close_position', 'subsequent_respect'],
            'timeframe_sensitivity': 'high'
        }
    
    def _get_mitigation_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['order_block_return', 'mitigation_level', 'percentage_filled'],
            'confidence_factors': ['mitigation_speed', 'fill_percentage', 'reaction_strength'],
            'timeframe_sensitivity': 'medium'
        }
    
    def _get_turtle_soup_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['false_breakout', 'quick_reversal', 'stop_hunt'],
            'confidence_factors': ['reversal_speed', 'reversal_distance', 'volume_spike'],
            'timeframe_sensitivity': 'high'
        }
    
    def _get_judas_swing_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['session_open', 'initial_move', 'reversal'],
            'confidence_factors': ['session_timing', 'move_size', 'reversal_strength'],
            'timeframe_sensitivity': 'high'
        }
    
    def _get_power_of_three_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['accumulation_phase', 'manipulation_phase', 'distribution_phase'],
            'confidence_factors': ['phase_clarity', 'manipulation_size', 'distribution_follow_through'],
            'timeframe_sensitivity': 'low'
        }
    
    def _get_smt_divergence_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['price_divergence', 'momentum_divergence', 'correlation_pairs'],
            'confidence_factors': ['divergence_strength', 'correlation_coefficient', 'timeframe_alignment'],
            'timeframe_sensitivity': 'medium'
        }
    
    def _calculate_template_match(self, pattern: Dict[str, Any], template: Dict[str, Any]) -> float:
        """Calculate how well a pattern matches its template"""
        # Simplified template matching
        base_confidence = pattern.get('confidence', 0.5)
        
        # Adjust based on template requirements
        if 'strength' in pattern and pattern['strength'] > 2:
            base_confidence *= 1.1
        
        if 'respect_count' in pattern and pattern['respect_count'] > 3:
            base_confidence *= 1.15
        
        return min(base_confidence, 1.0)
    
    async def _get_market_data(self, symbol: str, timeframe: TimeFrame, periods: int) -> pd.DataFrame:
        """Get real market data for pattern recognition"""
        try:
            # Import yfinance for real data
            import yfinance as yf
            
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
            
            # Calculate period based on periods requested
            if interval in ["1m", "5m"]:
                days = min(periods // 400, 7)  # Limited for intraday
                period = f"{max(1, days)}d"
            elif interval in ["15m", "30m"]:
                days = min(periods // 50, 30)
                period = f"{max(1, days)}d"
            elif interval == "1h":
                days = min(periods // 24, 30)
                period = f"{max(1, days)}d"
            elif interval == "4h":
                days = min(periods // 6, 60)
                period = f"{max(1, days)}d"
            else:
                days = min(periods, 365)
                period = f"{max(1, days)}d"
            
            # Get data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)
            
            if not hist.empty:
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
                
                # Limit to requested periods
                if len(data) > periods:
                    data = data.tail(periods)
                
                return data
            else:
                # Return error for invalid symbols rather than mock data
                raise ValueError(f"No market data available for {symbol}. Please verify symbol format.")
                
        except Exception as e:
            print(f"Error fetching market data for {symbol}: {str(e)}")
            # Return error instead of mock data
            raise ValueError(f"Cannot fetch real market data for {symbol}: {str(e)}")
    


    # Additional pattern detection methods for comprehensive ICT coverage
    
    def _detect_market_structure(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect market structure patterns (HH, HL, LH, LL)"""
        patterns = []
        
        for i in range(20, len(data) - 10):
            # Find swing highs and lows
            current_high = data['high'].iloc[i]
            current_low = data['low'].iloc[i]
            
            # Check if current high is higher than previous swings
            prev_highs = data['high'].iloc[i-10:i]
            next_highs = data['high'].iloc[i+1:i+6]
            
            if (current_high > prev_highs.max() and 
                current_high > next_highs.max()):
                
                # This is a swing high - check if it's higher than previous swing high
                prev_swing_high = prev_highs.max()
                if current_high > prev_swing_high:
                    patterns.append({
                        'timestamp': data.iloc[i]['timestamp'],
                        'price': current_high,
                        'type': 'higher_high',
                        'confidence': 0.8,
                        'previous_swing': prev_swing_high,
                        'status': 'confirmed'
                    })
        
        return patterns
    
    def _detect_liquidity_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect general liquidity patterns"""
        patterns = []
        
        # Find areas where price has repeatedly respected levels
        for i in range(10, len(data) - 5):
            current_level = data['high'].iloc[i]
            
            # Count how many times price respected this level
            respect_count = 0
            for j in range(max(0, i-20), min(len(data), i+10)):
                if abs(data['high'].iloc[j] - current_level) / current_level < 0.002:
                    respect_count += 1
            
            if respect_count >= 3:
                patterns.append({
                    'timestamp': data.iloc[i]['timestamp'],
                    'price': current_level,
                    'type': 'liquidity_level',
                    'respect_count': respect_count,
                    'confidence': min(respect_count / 5, 1.0),
                    'status': 'active'
                })
        
        return patterns
    
    def _detect_supply_demand_zones(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect supply and demand zones"""
        patterns = []
        
        for i in range(20, len(data) - 5):
            # Look for strong moves away from levels
            current_candle = data.iloc[i]
            
            # Strong bullish move (demand zone)
            if current_candle['close'] > current_candle['open']:
                body_size = current_candle['close'] - current_candle['open']
                avg_body = np.mean([abs(data['close'].iloc[j] - data['open'].iloc[j]) 
                                  for j in range(i-10, i)])
                
                if body_size > avg_body * 2:  # Strong bullish candle
                    # This creates a demand zone
                    patterns.append({
                        'timestamp': current_candle['timestamp'],
                        'zone_high': current_candle['high'],
                        'zone_low': current_candle['low'],
                        'type': 'demand_zone',
                        'strength': body_size / avg_body,
                        'confidence': min(body_size / avg_body / 3, 1.0),
                        'status': 'fresh'
                    })
        
        return patterns
    
    def _detect_premium_discount(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect premium/discount levels"""
        patterns = []
        
        if len(data) < 50:
            return patterns
        
        # Calculate current position in range
        range_high = data['high'].iloc[-50:].max()
        range_low = data['low'].iloc[-50:].min()
        current_price = data['close'].iloc[-1]
        
        if range_high > range_low:
            position = (current_price - range_low) / (range_high - range_low)
            
            if position > 0.7:  # Premium area
                patterns.append({
                    'timestamp': data.iloc[-1]['timestamp'],
                    'level': current_price,
                    'type': 'premium',
                    'position_in_range': position,
                    'confidence': min((position - 0.7) / 0.3, 1.0),
                    'range_high': range_high,
                    'range_low': range_low,
                    'status': 'active'
                })
            elif position < 0.3:  # Discount area
                patterns.append({
                    'timestamp': data.iloc[-1]['timestamp'],
                    'level': current_price,
                    'type': 'discount',
                    'position_in_range': position,
                    'confidence': min((0.3 - position) / 0.3, 1.0),
                    'range_high': range_high,
                    'range_low': range_low,
                    'status': 'active'
                })
        
        return patterns
    
    def _detect_dealing_ranges(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect dealing ranges (consolidation periods)"""
        patterns = []
        
        for i in range(30, len(data) - 10):
            # Look for periods of low volatility
            window = data.iloc[i-20:i]
            range_high = window['high'].max()
            range_low = window['low'].min()
            range_size = range_high - range_low
            
            # Calculate average true range for comparison
            atr_values = []
            for j in range(len(window)-1):
                high_low = window['high'].iloc[j] - window['low'].iloc[j]
                high_close = abs(window['high'].iloc[j] - window['close'].iloc[j-1]) if j > 0 else high_low
                low_close = abs(window['low'].iloc[j] - window['close'].iloc[j-1]) if j > 0 else high_low
                atr_values.append(max(high_low, high_close, low_close))
            
            avg_atr = np.mean(atr_values) if atr_values else range_size
            
            # If range is tight compared to ATR, it's a dealing range
            if range_size < avg_atr * 3:
                patterns.append({
                    'timestamp': data.iloc[i]['timestamp'],
                    'range_high': range_high,
                    'range_low': range_low,
                    'range_size': range_size,
                    'type': 'dealing_range',
                    'tightness': avg_atr / (range_size + 1e-10),
                    'confidence': min(avg_atr / (range_size + 1e-10) / 2, 1.0),
                    'status': 'active'
                })
        
        return patterns
    
    def _detect_swing_points(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect swing highs and lows"""
        patterns = []
        
        for i in range(10, len(data) - 10):
            current_high = data['high'].iloc[i]
            current_low = data['low'].iloc[i]
            
            # Check for swing high
            left_highs = data['high'].iloc[i-5:i]
            right_highs = data['high'].iloc[i+1:i+6]
            
            if (current_high > left_highs.max() and 
                current_high > right_highs.max()):
                patterns.append({
                    'timestamp': data.iloc[i]['timestamp'],
                    'price': current_high,
                    'type': 'swing_high',
                    'strength': (current_high - left_highs.max()) / current_high,
                    'confidence': 0.8,
                    'status': 'confirmed'
                })
            
            # Check for swing low
            left_lows = data['low'].iloc[i-5:i]
            right_lows = data['low'].iloc[i+1:i+6]
            
            if (current_low < left_lows.min() and 
                current_low < right_lows.min()):
                patterns.append({
                    'timestamp': data.iloc[i]['timestamp'],
                    'price': current_low,
                    'type': 'swing_low',
                    'strength': (left_lows.min() - current_low) / current_low,
                    'confidence': 0.8,
                    'status': 'confirmed'
                })
        
        return patterns
    
    def _detect_market_maker_models(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect market maker buy/sell model patterns"""
        patterns = []
        
        for i in range(30, len(data) - 10):
            # Look for accumulation -> manipulation -> distribution
            window = data.iloc[i-30:i]
            
            # Split into phases
            accumulation = window.iloc[:10]
            manipulation = window.iloc[10:20]
            distribution = window.iloc[20:]
            
            # Check for market maker sell model
            acc_range = accumulation['high'].max() - accumulation['low'].min()
            manip_high = manipulation['high'].max()
            dist_trend = (distribution['close'].iloc[-1] - distribution['close'].iloc[0]) / distribution['close'].iloc[0]
            
            if (manip_high > accumulation['high'].max() and dist_trend < -0.02):
                patterns.append({
                    'timestamp': data.iloc[i]['timestamp'],
                    'model_type': 'sell_model',
                    'accumulation_range': acc_range,
                    'manipulation_high': manip_high,
                    'distribution_decline': dist_trend,
                    'confidence': min(abs(dist_trend) * 10, 1.0),
                    'status': 'developing'
                })
        
        return patterns
    
    def _detect_optimal_trade_entry(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect optimal trade entry setups (62-79% retracements)"""
        patterns = []
        
        # Find recent swings
        swing_highs = []
        swing_lows = []
        
        for i in range(10, len(data) - 10):
            if (data['high'].iloc[i] > data['high'].iloc[i-5:i].max() and
                data['high'].iloc[i] > data['high'].iloc[i+1:i+6].max()):
                swing_highs.append((i, data['high'].iloc[i]))
            
            if (data['low'].iloc[i] < data['low'].iloc[i-5:i].min() and
                data['low'].iloc[i] < data['low'].iloc[i+1:i+6].min()):
                swing_lows.append((i, data['low'].iloc[i]))
        
        # Calculate retracement levels
        for i, (swing_idx, swing_price) in enumerate(swing_highs):
            if i > 0:
                prev_swing_idx, prev_swing_price = swing_highs[i-1]
                swing_range = abs(swing_price - prev_swing_price)
                
                # 62% and 79% retracement levels
                if swing_price > prev_swing_price:  # Uptrend
                    ote_62 = swing_price - (swing_range * 0.62)
                    ote_79 = swing_price - (swing_range * 0.79)
                    
                    current_price = data['close'].iloc[-1]
                    if ote_79 <= current_price <= ote_62:
                        patterns.append({
                            'timestamp': data.iloc[-1]['timestamp'],
                            'swing_high': swing_price,
                            'swing_low': prev_swing_price,
                            'ote_62_level': ote_62,
                            'ote_79_level': ote_79,
                            'current_retracement': (swing_price - current_price) / swing_range,
                            'type': 'bullish_ote',
                            'confidence': 0.85,
                            'status': 'active'
                        })
        
        return patterns
    
    def _detect_liquidity_voids(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect liquidity voids (price inefficiencies)"""
        patterns = []
        
        for i in range(5, len(data) - 5):
            # Look for single print areas (low volume zones)
            window = data.iloc[i-5:i+5]
            current_level = data['close'].iloc[i]
            
            # Count how often price traded at this level
            level_touches = 0
            tolerance = current_level * 0.001  # 0.1% tolerance
            
            for j in range(len(window)):
                if (abs(window['high'].iloc[j] - current_level) <= tolerance or
                    abs(window['low'].iloc[j] - current_level) <= tolerance or
                    abs(window['close'].iloc[j] - current_level) <= tolerance):
                    level_touches += 1
            
            # If price rarely traded at this level, it's a void
            if level_touches <= 2:
                patterns.append({
                    'timestamp': data.iloc[i]['timestamp'],
                    'void_level': current_level,
                    'touch_count': level_touches,
                    'type': 'liquidity_void',
                    'confidence': max(0, 1 - level_touches / 5),
                    'status': 'identified'
                })
        
        return patterns
    
    # Additional detection methods for time-based and advanced concepts
    
    def _detect_killzone_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect killzone-specific patterns"""
        patterns = []
        current_hour = datetime.utcnow().hour
        
        # London Killzone (7-10 UTC)
        if 7 <= current_hour <= 10:
            if len(data) >= 4:
                london_volatility = np.std(data['close'].iloc[-4:])
                avg_volatility = np.std(data['close'].iloc[-20:])
                
                if london_volatility > avg_volatility * 1.5:
                    patterns.append({
                        'timestamp': data.iloc[-1]['timestamp'],
                        'killzone': 'london',
                        'volatility_ratio': london_volatility / avg_volatility,
                        'type': 'high_activity_killzone',
                        'confidence': min(london_volatility / avg_volatility / 2, 1.0),
                        'status': 'active'
                    })
        
        # New York Killzone (13-16 UTC)
        elif 13 <= current_hour <= 16:
            if len(data) >= 4:
                ny_momentum = (data['close'].iloc[-1] - data['close'].iloc[-4]) / data['close'].iloc[-4]
                
                patterns.append({
                    'timestamp': data.iloc[-1]['timestamp'],
                    'killzone': 'new_york',
                    'momentum': ny_momentum,
                    'type': 'momentum_killzone',
                    'confidence': min(abs(ny_momentum) * 20, 1.0),
                    'status': 'active'
                })
        
        return patterns
    
    def _detect_session_open_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect session opening patterns"""
        patterns = []
        current_hour = datetime.utcnow().hour
        
        # Check if we're at a major session open
        if current_hour in [0, 7, 13]:  # Sydney, London, New York
            if len(data) >= 3:
                session_name = {0: 'sydney', 7: 'london', 13: 'new_york'}[current_hour]
                
                # Look for gap or strong directional move
                open_price = data['open'].iloc[-1]
                prev_close = data['close'].iloc[-2]
                gap_size = abs(open_price - prev_close) / prev_close
                
                if gap_size > 0.001:  # Significant gap
                    patterns.append({
                        'timestamp': data.iloc[-1]['timestamp'],
                        'session': session_name,
                        'gap_size': gap_size,
                        'gap_direction': 'up' if open_price > prev_close else 'down',
                        'type': 'session_gap',
                        'confidence': min(gap_size * 100, 1.0),
                        'status': 'fresh'
                    })
        
        return patterns
    
    def _detect_fibonacci_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Fibonacci-based patterns"""
        patterns = []
        
        # Find significant swings for Fibonacci analysis
        if len(data) < 20:
            return patterns
        
        swing_high = data['high'].iloc[-20:].max()
        swing_low = data['low'].iloc[-20:].min()
        current_price = data['close'].iloc[-1]
        
        if swing_high > swing_low:
            fib_range = swing_high - swing_low
            
            # Key Fibonacci levels
            fib_levels = {
                'fib_236': swing_high - (fib_range * 0.236),
                'fib_382': swing_high - (fib_range * 0.382),
                'fib_500': swing_high - (fib_range * 0.500),
                'fib_618': swing_high - (fib_range * 0.618),
                'fib_786': swing_high - (fib_range * 0.786)
            }
            
            # Check proximity to Fibonacci levels
            for level_name, level_price in fib_levels.items():
                distance = abs(current_price - level_price) / current_price
                
                if distance < 0.005:  # Within 0.5%
                    patterns.append({
                        'timestamp': data.iloc[-1]['timestamp'],
                        'fibonacci_level': level_name,
                        'level_price': level_price,
                        'distance_ratio': distance,
                        'swing_high': swing_high,
                        'swing_low': swing_low,
                        'type': 'fibonacci_confluence',
                        'confidence': max(0, 1 - distance * 100),
                        'status': 'active'
                    })
        
        return patterns
    
    def _detect_silver_bullet_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Silver Bullet setup patterns"""
        patterns = []
        current_hour = datetime.utcnow().hour
        
        # Silver Bullet is specific to 10-11 AM EST (15-16 UTC)
        if 15 <= current_hour <= 16:
            if len(data) >= 5:
                # Look for consolidation followed by breakout
                recent_data = data.iloc[-5:]
                range_size = recent_data['high'].max() - recent_data['low'].min()
                avg_range = np.mean(data['high'].iloc[-20:] - data['low'].iloc[-20:])
                
                if range_size < avg_range * 0.6:  # Tight consolidation
                    patterns.append({
                        'timestamp': data.iloc[-1]['timestamp'],
                        'setup_type': 'silver_bullet',
                        'consolidation_range': range_size,
                        'average_range': avg_range,
                        'compression_ratio': range_size / avg_range,
                        'type': 'consolidation_breakout_setup',
                        'confidence': max(0, 1 - (range_size / avg_range)),
                        'status': 'forming'
                    })
        
        return patterns
    
    def _detect_asian_range_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Asian range breakout patterns"""
        patterns = []
        current_hour = datetime.utcnow().hour
        
        # Asian session typically 20:00 - 06:00 UTC
        if 20 <= current_hour or current_hour <= 6:
            if len(data) >= 8:
                # Define Asian range (last 8 hours as proxy)
                asian_data = data.iloc[-8:]
                asian_high = asian_data['high'].max()
                asian_low = asian_data['low'].min()
                asian_range = asian_high - asian_low
                
                current_price = data['close'].iloc[-1]
                
                # Check for potential breakout
                upper_break_threshold = asian_high * 1.001
                lower_break_threshold = asian_low * 0.999
                
                patterns.append({
                    'timestamp': data.iloc[-1]['timestamp'],
                    'asian_high': asian_high,
                    'asian_low': asian_low,
                    'asian_range': asian_range,
                    'current_price': current_price,
                    'breakout_potential': 'high' if (current_price > upper_break_threshold or 
                                                   current_price < lower_break_threshold) else 'low',
                    'type': 'asian_range_setup',
                    'confidence': 0.7,
                    'status': 'monitoring'
                })
        
        return patterns
    
    def _detect_ny_reversal_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect New York reversal patterns"""
        patterns = []
        current_hour = datetime.utcnow().hour
        
        # NY session reversal typically around 16:00-17:00 UTC
        if 16 <= current_hour <= 17:
            if len(data) >= 6:
                # Look for trend exhaustion and reversal signals
                trend_data = data.iloc[-6:]
                
                # Calculate trend strength
                trend_direction = (trend_data['close'].iloc[-1] - trend_data['close'].iloc[0]) / trend_data['close'].iloc[0]
                
                # Look for reversal signals (long wicks, volume spikes)
                last_candle = trend_data.iloc[-1]
                candle_range = last_candle['high'] - last_candle['low']
                body_size = abs(last_candle['close'] - last_candle['open'])
                
                wick_ratio = (candle_range - body_size) / candle_range if candle_range > 0 else 0
                
                if wick_ratio > 0.6:  # Significant wicks indicating rejection
                    patterns.append({
                        'timestamp': data.iloc[-1]['timestamp'],
                        'trend_direction': 'bullish' if trend_direction > 0 else 'bearish',
                        'trend_strength': abs(trend_direction),
                        'wick_ratio': wick_ratio,
                        'reversal_signal': 'strong' if wick_ratio > 0.8 else 'moderate',
                        'type': 'ny_reversal_setup',
                        'confidence': min(wick_ratio * 1.2, 1.0),
                        'status': 'developing'
                    })
        
        return patterns
    
    def _detect_london_killzone_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect London killzone specific patterns"""
        patterns = []
        current_hour = datetime.utcnow().hour
        
        # London killzone 7-10 UTC
        if 7 <= current_hour <= 10:
            if len(data) >= 4:
                london_data = data.iloc[-4:]  # Last 4 hours
                
                # Calculate London momentum
                london_momentum = (london_data['close'].iloc[-1] - london_data['open'].iloc[0]) / london_data['open'].iloc[0]
                london_volatility = np.std(london_data['close'])
                
                # Look for strong directional moves
                if abs(london_momentum) > 0.005:  # Significant move
                    patterns.append({
                        'timestamp': data.iloc[-1]['timestamp'],
                        'momentum': london_momentum,
                        'volatility': london_volatility,
                        'direction': 'bullish' if london_momentum > 0 else 'bearish',
                        'strength': abs(london_momentum),
                        'type': 'london_directional_move',
                        'confidence': min(abs(london_momentum) * 50, 1.0),
                        'status': 'active'
                    })
        
        return patterns
    
    # Advanced concept detection methods
    
    def _detect_liquidity_run_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect liquidity run patterns"""
        patterns = []
        
        for i in range(20, len(data) - 5):
            # Find equal highs/lows that might get swept
            current_level = data['high'].iloc[i]
            
            # Look for similar levels nearby
            similar_levels = []
            for j in range(max(0, i-15), min(len(data), i+5)):
                if abs(data['high'].iloc[j] - current_level) / current_level < 0.002:
                    similar_levels.append(j)
            
            if len(similar_levels) >= 2:
                # Check if level gets swept later
                swept = False
                for k in range(i+1, min(i+10, len(data))):
                    if data['high'].iloc[k] > current_level * 1.001:
                        swept = True
                        break
                
                patterns.append({
                    'timestamp': data.iloc[i]['timestamp'],
                    'liquidity_level': current_level,
                    'similar_level_count': len(similar_levels),
                    'swept': swept,
                    'type': 'liquidity_run',
                    'confidence': min(len(similar_levels) / 3, 1.0),
                    'status': 'swept' if swept else 'pending'
                })
        
        return patterns
    
    def _detect_accumulation_distribution(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect accumulation/distribution patterns"""
        patterns = []
        
        if len(data) < 30:
            return patterns
        
        # Analyze in chunks to identify phases
        for i in range(30, len(data), 10):
            window = data.iloc[i-30:i]
            
            # Calculate volume-price relationship
            volume_trend = np.polyfit(range(len(window)), window['volume'], 1)[0]
            price_trend = np.polyfit(range(len(window)), window['close'], 1)[0]
            
            # Accumulation: increasing volume, sideways/up price
            # Distribution: increasing volume, sideways/down price
            
            if volume_trend > 0:  # Increasing volume
                if price_trend > 0:
                    phase = 'accumulation'
                    confidence = min(volume_trend * price_trend * 1000, 1.0)
                elif price_trend < 0:
                    phase = 'distribution'
                    confidence = min(volume_trend * abs(price_trend) * 1000, 1.0)
                else:
                    phase = 'preparation'
                    confidence = 0.5
                
                patterns.append({
                    'timestamp': data.iloc[i-1]['timestamp'],
                    'phase': phase,
                    'volume_trend': volume_trend,
                    'price_trend': price_trend,
                    'type': 'accumulation_distribution',
                    'confidence': confidence,
                    'status': 'identified'
                })
        
        return patterns
    
    def _detect_order_flow_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect order flow patterns"""
        patterns = []
        
        for i in range(5, len(data)):
            # Analyze order flow using close position within range
            window = data.iloc[i-5:i]
            
            order_flow_scores = []
            for j in range(len(window)):
                candle = window.iloc[j]
                candle_range = candle['high'] - candle['low']
                
                if candle_range > 0:
                    close_position = (candle['close'] - candle['low']) / candle_range
                    order_flow_scores.append(close_position)
            
            if order_flow_scores:
                avg_order_flow = np.mean(order_flow_scores)
                
                # Strong bullish order flow if consistently closing near highs
                if avg_order_flow > 0.7:
                    flow_type = 'bullish'
                    strength = avg_order_flow
                elif avg_order_flow < 0.3:
                    flow_type = 'bearish'
                    strength = 1 - avg_order_flow
                else:
                    flow_type = 'neutral'
                    strength = 0.5
                
                patterns.append({
                    'timestamp': data.iloc[i]['timestamp'],
                    'order_flow_type': flow_type,
                    'strength': strength,
                    'avg_close_position': avg_order_flow,
                    'type': 'order_flow_analysis',
                    'confidence': abs(avg_order_flow - 0.5) * 2,
                    'status': 'measured'
                })
        
        return patterns
    
    def _detect_ipda_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect Interbank Price Delivery Algorithm patterns"""
        patterns = []
        
        # IPDA looks for algorithmic price delivery patterns
        for i in range(20, len(data) - 5):
            window = data.iloc[i-20:i]
            
            # Look for systematic price movements
            price_changes = window['close'].diff().dropna()
            
            # Calculate autocorrelation to detect systematic patterns
            if len(price_changes) > 10:
                # Simple pattern detection - look for recurring cycles
                autocorr_1 = np.corrcoef(price_changes[:-1], price_changes[1:])[0, 1]
                autocorr_5 = np.corrcoef(price_changes[:-5], price_changes[5:])[0, 1] if len(price_changes) > 5 else 0
                
                if not np.isnan(autocorr_1) and abs(autocorr_1) > 0.3:
                    patterns.append({
                        'timestamp': data.iloc[i]['timestamp'],
                        'pattern_strength': abs(autocorr_1),
                        'cycle_type': 'short_term' if abs(autocorr_1) > abs(autocorr_5) else 'medium_term',
                        'autocorr_1': autocorr_1,
                        'autocorr_5': autocorr_5,
                        'type': 'ipda_algorithm',
                        'confidence': min(abs(autocorr_1), 1.0),
                        'status': 'detected'
                    })
        
        return patterns
    
    def _detect_algo_delivery_patterns(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect algorithmic price delivery patterns"""
        patterns = []
        
        # Look for systematic delivery patterns in price action
        for i in range(30, len(data)):
            window = data.iloc[i-30:i]
            
            # Calculate delivery efficiency (how smoothly price moves)
            price_path = window['close'].values
            direct_distance = abs(price_path[-1] - price_path[0])
            actual_distance = np.sum(np.abs(np.diff(price_path)))
            
            if actual_distance > 0:
                efficiency = direct_distance / actual_distance
                
                # High efficiency suggests algorithmic delivery
                if efficiency > 0.7:
                    patterns.append({
                        'timestamp': data.iloc[i]['timestamp'],
                        'delivery_efficiency': efficiency,
                        'direct_distance': direct_distance,
                        'actual_distance': actual_distance,
                        'delivery_type': 'efficient' if efficiency > 0.8 else 'moderate',
                        'type': 'algo_price_delivery',
                        'confidence': efficiency,
                        'status': 'identified'
                    })
        
        return patterns
    
    def _detect_generic_pattern(self, data: pd.DataFrame, pattern_name: str) -> List[Dict[str, Any]]:
        """Generic pattern detection for any missing patterns"""
        patterns = []
        
        # Fallback detection based on volatility and volume
        if len(data) >= 10:
            recent_volatility = np.std(data['close'].iloc[-10:])
            avg_volatility = np.std(data['close'])
            
            if recent_volatility > avg_volatility * 1.5:
                patterns.append({
                    'timestamp': data.iloc[-1]['timestamp'],
                    'pattern_name': pattern_name,
                    'volatility_ratio': recent_volatility / avg_volatility,
                    'type': 'generic_pattern',
                    'confidence': min((recent_volatility / avg_volatility - 1) / 2, 1.0),
                    'status': 'detected'
                })
        
        return patterns

    # Template methods for all ICT concepts
    
    def _get_market_structure_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['swing_points', 'trend_direction', 'structure_breaks'],
            'confidence_factors': ['swing_strength', 'trend_consistency', 'volume_confirmation'],
            'timeframe_sensitivity': 'medium'
        }
    
    def _get_liquidity_pools_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['equal_levels', 'level_count', 'sweep_behavior'],
            'confidence_factors': ['level_precision', 'touch_count', 'sweep_reaction'],
            'timeframe_sensitivity': 'low'
        }
    
    def _get_supply_demand_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['strong_move', 'zone_formation', 'reaction_strength'],
            'confidence_factors': ['move_size', 'volume_spike', 'zone_respect'],
            'timeframe_sensitivity': 'medium'
        }
    
    def _get_premium_discount_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['range_position', 'fibonacci_levels', 'price_action'],
            'confidence_factors': ['range_clarity', 'level_confluence', 'reaction_strength'],
            'timeframe_sensitivity': 'high'
        }
    
    def _get_dealing_ranges_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['consolidation', 'range_boundaries', 'breakout_potential'],
            'confidence_factors': ['range_tightness', 'time_duration', 'volume_contraction'],
            'timeframe_sensitivity': 'low'
        }
    
    def _get_swing_points_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['local_extremes', 'confirmation_candles', 'significance'],
            'confidence_factors': ['extremity_level', 'confirmation_strength', 'volume'],
            'timeframe_sensitivity': 'medium'
        }
    
    def _get_market_maker_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['accumulation', 'manipulation', 'distribution'],
            'confidence_factors': ['phase_clarity', 'volume_profile', 'price_action'],
            'timeframe_sensitivity': 'low'
        }
    
    def _get_ote_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['swing_identification', 'retracement_level', 'confluence'],
            'confidence_factors': ['fibonacci_precision', 'multiple_confirmations', 'reaction'],
            'timeframe_sensitivity': 'high'
        }
    
    def _get_liquidity_voids_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['price_inefficiency', 'volume_gap', 'single_prints'],
            'confidence_factors': ['void_size', 'time_duration', 'fill_probability'],
            'timeframe_sensitivity': 'high'
        }
    
    # Time & Price Theory templates
    
    def _get_killzones_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['time_window', 'volatility_expansion', 'directional_bias'],
            'confidence_factors': ['session_strength', 'volume_increase', 'trend_continuation'],
            'timeframe_sensitivity': 'high'
        }
    
    def _get_session_opens_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['gap_formation', 'initial_direction', 'follow_through'],
            'confidence_factors': ['gap_size', 'volume_confirmation', 'momentum'],
            'timeframe_sensitivity': 'high'
        }
    
    def _get_fibonacci_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['swing_identification', 'level_confluence', 'price_reaction'],
            'confidence_factors': ['level_precision', 'multiple_touches', 'volume_reaction'],
            'timeframe_sensitivity': 'medium'
        }
    
    # Strategy-specific templates
    
    def _get_silver_bullet_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['time_window', 'consolidation', 'breakout_setup'],
            'confidence_factors': ['compression_ratio', 'volume_buildup', 'directional_bias'],
            'timeframe_sensitivity': 'high'
        }
    
    def _get_asian_range_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['range_formation', 'breakout_potential', 'session_timing'],
            'confidence_factors': ['range_clarity', 'time_duration', 'volume_pattern'],
            'timeframe_sensitivity': 'medium'
        }
    
    def _get_ny_reversal_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['trend_exhaustion', 'reversal_signals', 'session_timing'],
            'confidence_factors': ['trend_strength', 'reversal_confirmation', 'volume_divergence'],
            'timeframe_sensitivity': 'high'
        }
    
    def _get_london_killzone_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['session_timing', 'volatility_expansion', 'directional_move'],
            'confidence_factors': ['momentum_strength', 'volume_confirmation', 'follow_through'],
            'timeframe_sensitivity': 'high'
        }
    
    # Additional template methods for remaining concepts
    def _get_fvg_sniper_template(self) -> Dict[str, Any]:
        return self._get_fvg_template()
    
    def _get_ob_refined_template(self) -> Dict[str, Any]:
        return self._get_order_block_template()
    
    def _get_breaker_refined_template(self) -> Dict[str, Any]:
        return self._get_breaker_template()
    
    def _get_rejection_refined_template(self) -> Dict[str, Any]:
        return self._get_rejection_template()
    
    def _get_smt_refined_template(self) -> Dict[str, Any]:
        return self._get_smt_divergence_template()
    
    def _get_turtle_soup_refined_template(self) -> Dict[str, Any]:
        return self._get_turtle_soup_template()
    
    def _get_power_three_refined_template(self) -> Dict[str, Any]:
        return self._get_power_of_three_template()
    
    def _get_daily_bias_raid_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['daily_bias', 'liquidity_raid', 'reversal_setup'],
            'confidence_factors': ['bias_strength', 'raid_completion', 'reversal_signals'],
            'timeframe_sensitivity': 'medium'
        }
    
    def _get_am_session_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['morning_bias', 'session_open', 'directional_move'],
            'confidence_factors': ['bias_clarity', 'momentum_strength', 'volume_confirmation'],
            'timeframe_sensitivity': 'high'
        }
    
    def _get_pm_session_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['afternoon_reversal', 'session_change', 'trend_exhaustion'],
            'confidence_factors': ['reversal_strength', 'timing_precision', 'volume_divergence'],
            'timeframe_sensitivity': 'high'
        }
    
    def _get_ote_refined_template(self) -> Dict[str, Any]:
        return self._get_ote_template()
    
    # Placeholder templates for remaining concepts
    def _get_range_expectations_template(self) -> Dict[str, Any]:
        return self._get_dealing_ranges_template()
    
    def _get_session_raids_template(self) -> Dict[str, Any]:
        return self._get_liquidity_runs_template()
    
    def _get_weekly_profiles_template(self) -> Dict[str, Any]:
        return self._get_market_structure_template()
    
    def _get_daily_bias_template(self) -> Dict[str, Any]:
        return self._get_market_structure_template()
    
    def _get_weekly_bias_template(self) -> Dict[str, Any]:
        return self._get_market_structure_template()
    
    def _get_monthly_bias_template(self) -> Dict[str, Any]:
        return self._get_market_structure_template()
    
    def _get_time_of_day_template(self) -> Dict[str, Any]:
        return self._get_killzones_template()
    
    def _get_high_prob_template(self) -> Dict[str, Any]:
        return self._get_ote_template()
    
    def _get_liquidity_runs_template(self) -> Dict[str, Any]:
        return self._get_liquidity_template()
    
    def _get_reversals_template(self) -> Dict[str, Any]:
        return self._get_market_structure_template()
    
    def _get_amd_template(self) -> Dict[str, Any]:
        return self._get_market_maker_template()
    
    def _get_order_flow_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['volume_analysis', 'price_action', 'flow_direction'],
            'confidence_factors': ['flow_strength', 'consistency', 'volume_confirmation'],
            'timeframe_sensitivity': 'high'
        }
    
    def _get_high_low_day_template(self) -> Dict[str, Any]:
        return self._get_market_structure_template()
    
    def _get_range_expansion_template(self) -> Dict[str, Any]:
        return self._get_dealing_ranges_template()
    
    def _get_inside_outside_template(self) -> Dict[str, Any]:
        return self._get_market_structure_template()
    
    def _get_weekly_advanced_template(self) -> Dict[str, Any]:
        return self._get_weekly_profiles_template()
    
    def _get_ipda_template(self) -> Dict[str, Any]:
        return {
            'required_features': ['algorithmic_patterns', 'systematic_delivery', 'price_efficiency'],
            'confidence_factors': ['pattern_consistency', 'delivery_precision', 'timing_accuracy'],
            'timeframe_sensitivity': 'low'
        }
    
    def _get_algo_delivery_template(self) -> Dict[str, Any]:
        return self._get_ipda_template()