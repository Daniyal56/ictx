import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import cv2

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
        
        # Pattern templates
        self.pattern_templates = {
            'order_block': self._get_order_block_template(),
            'fair_value_gap': self._get_fvg_template(),
            'breaker_block': self._get_breaker_template(),
            'liquidity_pool': self._get_liquidity_template(),
            'rejection_block': self._get_rejection_template(),
            'mitigation_block': self._get_mitigation_template(),
            'turtle_soup': self._get_turtle_soup_template(),
            'judas_swing': self._get_judas_swing_template(),
            'power_of_three': self._get_power_of_three_template(),
            'smt_divergence': self._get_smt_divergence_template()
        }
    
    async def detect_patterns(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        lookback_periods: int = 100
    ) -> List[Dict[str, Any]]:
        """Detect ICT patterns in market data"""
        
        # Get market data
        data = await self._get_market_data(symbol, timeframe, lookback_periods)
        
        detected_patterns = []
        
        # Detect each pattern type
        for pattern_name, template in self.pattern_templates.items():
            patterns = await self._detect_pattern_type(data, pattern_name, template)
            detected_patterns.extend(patterns)
        
        # Sort by confidence and recency
        detected_patterns.sort(key=lambda x: (x['confidence'], x['timestamp']), reverse=True)
        
        return detected_patterns[:20]  # Return top 20 patterns
    
    async def _detect_pattern_type(
        self, 
        data: pd.DataFrame, 
        pattern_name: str, 
        template: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Detect specific pattern type in data"""
        
        patterns = []
        
        if pattern_name == 'order_block':
            patterns = self._detect_order_blocks(data)
        elif pattern_name == 'fair_value_gap':
            patterns = self._detect_fair_value_gaps(data)
        elif pattern_name == 'breaker_block':
            patterns = self._detect_breaker_blocks(data)
        elif pattern_name == 'liquidity_pool':
            patterns = self._detect_liquidity_pools(data)
        elif pattern_name == 'rejection_block':
            patterns = self._detect_rejection_blocks(data)
        elif pattern_name == 'mitigation_block':
            patterns = self._detect_mitigation_blocks(data)
        elif pattern_name == 'turtle_soup':
            patterns = self._detect_turtle_soup(data)
        elif pattern_name == 'judas_swing':
            patterns = self._detect_judas_swing(data)
        elif pattern_name == 'power_of_three':
            patterns = self._detect_power_of_three(data)
        elif pattern_name == 'smt_divergence':
            patterns = self._detect_smt_divergence(data)
        
        # Add pattern name to each detection
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
        """Get market data for pattern recognition"""
        # Mock data for development - replace with real data source
        dates = pd.date_range(end=datetime.utcnow(), periods=periods, freq='H')
        np.random.seed(42)
        
        base_price = 1.1000
        prices = []
        
        for i in range(len(dates)):
            change = np.random.normal(0, 0.001)
            if i == 0:
                price = base_price
            else:
                price = prices[-1] + change
            prices.append(max(price, 0.5))  # Ensure positive prices
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p + abs(np.random.normal(0, 0.0005)) for p in prices],
            'low': [p - abs(np.random.normal(0, 0.0005)) for p in prices],
            'close': [p + np.random.normal(0, 0.0002) for p in prices],
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        return data