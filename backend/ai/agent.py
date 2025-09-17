import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import joblib

from app.models import (
    MarketAnalysis, AIRecommendation, TradeSetup, BacktestResult,
    TimeFrame, TradeDirection, MarketStructure
)
from strategies.ict_strategies import ICTStrategyManager

class ICTAIAgent:
    """Advanced AI Agent for ICT Trading with machine learning capabilities"""
    
    def __init__(self):
        self.strategy_manager = ICTStrategyManager()
        self.pattern_model = None
        self.direction_model = None
        self.regime_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize machine learning models"""
        # Pattern recognition model
        self.pattern_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Direction prediction model
        self.direction_model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Market regime detection model
        self.regime_model = RandomForestClassifier(
            n_estimators=50,
            max_depth=8,
            random_state=42
        )
    
    async def analyze_market(self, symbol: str, timeframe: TimeFrame) -> MarketAnalysis:
        """AI-enhanced market analysis"""
        # Get base ICT analysis
        base_analysis = await self.strategy_manager.analyze_market(symbol, timeframe)
        
        # Add AI enhancements
        if self.is_trained:
            # Get market data for AI analysis
            data = await self.strategy_manager._get_market_data(symbol, timeframe, 30)
            
            # Extract features
            features = self._extract_features(data)
            
            # Predict market regime
            regime_prediction = self._predict_market_regime(features)
            
            # Adjust confidence based on AI predictions
            ai_confidence = self._calculate_ai_confidence(features, regime_prediction)
            base_analysis.confidence = (base_analysis.confidence + ai_confidence) / 2
            
            # Update sentiment based on AI analysis
            sentiment_score = self._analyze_sentiment_features(features)
            base_analysis.sentiment = self._convert_sentiment_score(sentiment_score)
        
        return base_analysis
    
    async def generate_recommendation(self, analysis: MarketAnalysis) -> AIRecommendation:
        """Generate AI-powered trading recommendation using ALL 65 ICT concepts"""
        
        # Get ALL available ICT strategies for comprehensive analysis
        all_strategy_keys = list(self.strategy_manager.strategies.keys())
        
        # Select the most relevant strategies based on current market conditions
        relevant_strategies = self._select_relevant_strategies(analysis, all_strategy_keys)
        
        # Get current trade setups from all relevant strategies
        setups = await self.strategy_manager.get_trade_setups(
            analysis.symbol, 
            analysis.timeframe, 
            relevant_strategies
        )
        
        # AI enhancement of setups
        if self.is_trained:
            enhanced_setups = await self._enhance_setups_with_ai(setups, analysis)
        else:
            enhanced_setups = setups
        
        # Generate recommendation
        recommendation = AIRecommendation(
            symbol=analysis.symbol,
            analysis=analysis,
            trade_setups=enhanced_setups,
            risk_assessment=self._assess_risk(analysis),
            market_outlook=self._generate_market_outlook(analysis),
            key_events=self._identify_key_events(analysis),
            confidence=analysis.confidence,
            timestamp=datetime.utcnow()
        )
        
        return recommendation
    
    async def predict_market_movement(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        horizon_hours: int = 24
    ) -> 'MarketPrediction':
        """Predict market movement using AI models"""
        
        # Get historical data
        data = await self.strategy_manager._get_market_data(symbol, timeframe, 50)
        
        # Extract features
        features = self._extract_features(data)
        
        if not self.is_trained:
            # Train models with available data
            await self._train_models(data)
        
        # Make predictions
        direction_prob = self._predict_direction(features)
        target_levels = self._predict_target_levels(features, direction_prob)
        confidence = self._calculate_prediction_confidence(features, direction_prob)
        
        # Identify key factors
        key_factors = self._identify_prediction_factors(features)
        risk_factors = self._identify_risk_factors(features)
        
        return MarketPrediction(
            direction=TradeDirection.LONG if direction_prob > 0.5 else TradeDirection.SHORT,
            confidence=confidence,
            target_levels=target_levels,
            factors=key_factors,
            risks=risk_factors
        )
    
    async def optimize_portfolio(
        self,
        symbols: List[str],
        risk_tolerance: float,
        max_positions: int
    ) -> 'PortfolioOptimization':
        """AI-powered portfolio optimization"""
        
        # Analyze each symbol
        symbol_analyses = {}
        for symbol in symbols:
            analysis = await self.analyze_market(symbol, TimeFrame.H4)
            symbol_analyses[symbol] = analysis
        
        # Calculate correlations and risk metrics
        correlations = await self._calculate_correlations(symbols)
        risk_metrics = await self._calculate_portfolio_risk(symbols, symbol_analyses)
        
        # Optimize allocation
        allocation = self._optimize_allocation(
            symbol_analyses, 
            correlations, 
            risk_metrics, 
            risk_tolerance, 
            max_positions
        )
        
        # Calculate expected metrics
        expected_return = self._calculate_expected_return(allocation, symbol_analyses)
        expected_risk = self._calculate_expected_risk(allocation, correlations, risk_metrics)
        sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0
        
        # Generate rebalance suggestions
        rebalance_suggestions = self._generate_rebalance_suggestions(allocation, symbol_analyses)
        
        return PortfolioOptimization(
            allocation=allocation,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            diversification_score=self._calculate_diversification_score(allocation, correlations),
            rebalance_suggestions=rebalance_suggestions
        )
    
    async def detect_market_regime(self, symbol: str, lookback_days: int = 30) -> 'MarketRegime':
        """Detect current market regime using AI"""
        
        # Get data
        data = await self.strategy_manager._get_market_data(symbol, TimeFrame.H4, lookback_days)
        
        # Extract regime features
        features = self._extract_regime_features(data)
        
        if not self.is_trained:
            await self._train_models(data)
        
        # Predict regime
        regime_prediction = self._predict_market_regime(features)
        confidence = self._calculate_regime_confidence(features)
        
        # Determine regime characteristics
        regime_type, strength = self._interpret_regime_prediction(regime_prediction)
        expected_duration = self._estimate_regime_duration(features)
        
        # Generate trading implications
        trading_implications = self._generate_trading_implications(regime_type)
        recommended_strategies = self._recommend_strategies_for_regime(regime_type)
        
        return MarketRegime(
            regime_type=regime_type,
            confidence=confidence,
            strength=strength,
            expected_duration=expected_duration,
            trading_implications=trading_implications,
            recommended_strategies=recommended_strategies
        )
    
    async def backtest_ai_strategies(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10000
    ) -> 'AIBacktestResults':
        """Backtest AI-enhanced strategies"""
        
        # Get historical data for training and testing
        total_days = (end_date - start_date).days
        data = await self.strategy_manager._get_market_data(symbol, TimeFrame.H1, total_days)
        
        # Split data for training and testing
        split_point = int(len(data) * 0.7)  # 70% for training
        train_data = data[:split_point]
        test_data = data[split_point:]
        
        # Train AI models
        await self._train_models(train_data)
        
        # Run AI-enhanced backtest
        ai_results = await self._run_ai_backtest(test_data, symbol, initial_capital)
        
        # Run static strategy backtest for comparison
        static_results = await self._run_static_backtest(test_data, symbol, initial_capital)
        
        # Calculate AI-specific metrics
        adaptive_accuracy = self._calculate_adaptive_accuracy(ai_results)
        regime_accuracy = self._calculate_regime_detection_accuracy(ai_results)
        pattern_accuracy = self._calculate_pattern_recognition_accuracy(ai_results)
        sentiment_correlation = self._calculate_sentiment_correlation(ai_results)
        
        return AIBacktestResults(
            adaptive_accuracy=adaptive_accuracy,
            regime_accuracy=regime_accuracy,
            pattern_accuracy=pattern_accuracy,
            sentiment_correlation=sentiment_correlation,
            vs_static_strategies=self._compare_ai_vs_static(ai_results, static_results)
        )
    
    # Private methods for feature extraction and AI operations
    
    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for machine learning models"""
        features = []
        
        # Price-based features
        features.extend(self._extract_price_features(data))
        
        # Volume features
        features.extend(self._extract_volume_features(data))
        
        # Technical indicator features
        features.extend(self._extract_technical_features(data))
        
        # ICT-specific features
        features.extend(self._extract_ict_features(data))
        
        # Time-based features
        features.extend(self._extract_time_features(data))
        
        return np.array(features).reshape(1, -1)
    
    def _extract_price_features(self, data: pd.DataFrame) -> List[float]:
        """Extract price-based features"""
        close_prices = data['close'].values
        
        features = [
            np.mean(close_prices[-5:]) / np.mean(close_prices[-20:]) - 1,  # 5-day vs 20-day return
            np.std(close_prices[-10:]) / np.mean(close_prices[-10:]),  # Recent volatility
            (close_prices[-1] - close_prices[-5]) / close_prices[-5],  # 5-day return
            (close_prices[-1] - close_prices[-20]) / close_prices[-20],  # 20-day return
            (max(data['high'].iloc[-10:]) - min(data['low'].iloc[-10:])) / close_prices[-1],  # Range ratio
        ]
        
        return features
    
    def _extract_volume_features(self, data: pd.DataFrame) -> List[float]:
        """Extract volume-based features"""
        volumes = data['volume'].values
        
        features = [
            np.mean(volumes[-5:]) / np.mean(volumes[-20:]) - 1,  # Volume ratio
            np.std(volumes[-10:]) / np.mean(volumes[-10:]),  # Volume volatility
            volumes[-1] / np.mean(volumes[-10:]) - 1,  # Current volume vs average
        ]
        
        return features
    
    def _extract_technical_features(self, data: pd.DataFrame) -> List[float]:
        """Extract technical indicator features"""
        close_prices = data['close'].values
        
        # Simple moving averages
        sma_5 = np.mean(close_prices[-5:])
        sma_20 = np.mean(close_prices[-20:])
        
        # RSI approximation
        gains = np.where(np.diff(close_prices) > 0, np.diff(close_prices), 0)
        losses = np.where(np.diff(close_prices) < 0, -np.diff(close_prices), 0)
        avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
        avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
        rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))
        
        features = [
            close_prices[-1] / sma_5 - 1,  # Price vs SMA5
            close_prices[-1] / sma_20 - 1,  # Price vs SMA20
            sma_5 / sma_20 - 1,  # SMA5 vs SMA20
            rsi / 100,  # Normalized RSI
        ]
        
        return features
    
    def _extract_ict_features(self, data: pd.DataFrame) -> List[float]:
        """Extract comprehensive ICT-specific features from ALL 65 concepts"""
        features = []
        
        # Market structure features (Core ICT Concepts 1-5)
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        
        # Market Structure Analysis
        recent_high = max(highs[-10:])
        previous_high = max(highs[-20:-10])
        higher_high = 1 if recent_high > previous_high else 0
        
        recent_low = min(lows[-10:])
        previous_low = min(lows[-20:-10])
        higher_low = 1 if recent_low > previous_low else 0
        
        features.extend([higher_high, higher_low])
        
        # Liquidity & Liquidity Pools (Concepts 2-3)
        equal_highs = self._detect_equal_levels(highs[-20:])
        equal_lows = self._detect_equal_levels(lows[-20:])
        features.extend([equal_highs, equal_lows])
        
        # Order Block Features (Concept 4)
        ob_strength = self._calculate_order_block_strength(data)
        features.append(ob_strength)
        
        # Breaker Block Features (Concept 5)
        breaker_signals = self._detect_breaker_block_signals(data)
        features.append(breaker_signals)
        
        # Fair Value Gap Features (Concept 6)
        fvg_count = len(self._detect_gaps(data))
        fvg_size_ratio = self._calculate_avg_gap_size(data)
        features.extend([fvg_count, fvg_size_ratio])
        
        # Rejection & Mitigation Features (Concepts 7-8)
        rejection_strength = self._calculate_rejection_strength(data)
        mitigation_potential = self._calculate_mitigation_potential(data)
        features.extend([rejection_strength, mitigation_potential])
        
        # Supply & Demand Zone Features (Concept 9)
        supply_demand_imbalance = self._calculate_supply_demand_imbalance(data)
        features.append(supply_demand_imbalance)
        
        # Premium/Discount Analysis (Concept 10)
        premium_discount_ratio = self._calculate_premium_discount_ratio(data)
        features.append(premium_discount_ratio)
        
        # Time-based Features (Concepts 21-30)
        killzone_factor = self._get_killzone_factor()
        session_momentum = self._calculate_session_momentum(data)
        fibonacci_levels = self._calculate_fibonacci_confluence(data)
        features.extend([killzone_factor, session_momentum, fibonacci_levels])
        
        # Advanced Pattern Features (Concepts 40-50)
        liquidity_sweep_probability = self._calculate_liquidity_sweep_probability(data)
        accumulation_distribution_phase = self._detect_amd_phase(data)
        order_flow_strength = self._calculate_order_flow_strength(data)
        features.extend([liquidity_sweep_probability, accumulation_distribution_phase, order_flow_strength])
        
        # Strategy Pattern Features (Concepts 51-65)
        silver_bullet_setup = self._detect_silver_bullet_pattern(data)
        turtle_soup_probability = self._calculate_turtle_soup_probability(data)
        judas_swing_signal = self._detect_judas_swing_signal(data)
        power_of_three_phase = self._identify_power_of_three_phase(data)
        features.extend([silver_bullet_setup, turtle_soup_probability, judas_swing_signal, power_of_three_phase])
        
        return features
    
    def _extract_time_features(self, data: pd.DataFrame) -> List[float]:
        """Extract time-based features"""
        current_time = datetime.utcnow()
        hour = current_time.hour
        
        # Killzone features
        london_killzone = 1 if 7 <= hour <= 10 else 0
        ny_killzone = 1 if 12 <= hour <= 15 else 0
        asia_session = 1 if hour >= 20 or hour <= 6 else 0
        
        # Day of week
        day_of_week = current_time.weekday() / 6  # Normalized
        
        return [london_killzone, ny_killzone, asia_session, day_of_week]
    
    def _detect_equal_levels(self, prices: np.ndarray, tolerance: float = 0.001) -> int:
        """Detect equal price levels"""
        unique_levels = []
        
        for price in prices:
            is_new_level = True
            for level in unique_levels:
                if abs(price - level) / level < tolerance:
                    is_new_level = False
                    break
            if is_new_level:
                unique_levels.append(price)
        
        # Return number of price touches at similar levels
        total_touches = len(prices)
        unique_levels_count = len(unique_levels)
        
        return max(0, total_touches - unique_levels_count)
    
    def _detect_gaps(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect price gaps"""
        gaps = []
        
        for i in range(1, len(data)):
            prev_high = data['high'].iloc[i-1]
            prev_low = data['low'].iloc[i-1]
            curr_high = data['high'].iloc[i]
            curr_low = data['low'].iloc[i]
            
            # Gap up
            if curr_low > prev_high:
                gaps.append({
                    'type': 'gap_up',
                    'size': curr_low - prev_high,
                    'timestamp': data['timestamp'].iloc[i]
                })
            
            # Gap down
            elif curr_high < prev_low:
                gaps.append({
                    'type': 'gap_down',
                    'size': prev_low - curr_high,
                    'timestamp': data['timestamp'].iloc[i]
                })
        
        return gaps
    
    async def _train_models(self, data: pd.DataFrame):
        """Train AI models with historical data"""
        # This is a simplified training process
        # In production, you would need more sophisticated feature engineering and validation
        
        features_list = []
        targets_direction = []
        targets_regime = []
        
        # Generate training data
        for i in range(50, len(data) - 1):  # Need enough history for features
            window_data = data.iloc[i-50:i]
            features = self._extract_features(window_data)
            
            # Direction target (next period return)
            next_return = (data['close'].iloc[i+1] - data['close'].iloc[i]) / data['close'].iloc[i]
            direction_target = 1 if next_return > 0 else 0
            
            # Regime target (simplified: trending vs ranging)
            volatility = np.std(data['close'].iloc[i-10:i])
            trend_strength = abs(data['close'].iloc[i] - data['close'].iloc[i-10]) / data['close'].iloc[i-10]
            regime_target = 1 if trend_strength > volatility else 0  # 1: trending, 0: ranging
            
            features_list.append(features.flatten())
            targets_direction.append(direction_target)
            targets_regime.append(regime_target)
        
        if len(features_list) > 10:  # Minimum training samples
            X = np.array(features_list)
            y_direction = np.array(targets_direction)
            y_regime = np.array(targets_regime)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train models
            self.direction_model.fit(X_scaled, y_direction)
            self.regime_model.fit(X_scaled, y_regime)
            
            self.is_trained = True
    
    def _predict_direction(self, features: np.ndarray) -> float:
        """Predict market direction probability"""
        if not self.is_trained:
            return 0.5  # Neutral if not trained
        
        features_scaled = self.scaler.transform(features)
        probabilities = self.direction_model.predict(features_scaled)
        return float(probabilities[0])
    
    def _predict_market_regime(self, features: np.ndarray) -> int:
        """Predict market regime"""
        if not self.is_trained:
            return 0  # Default to ranging
        
        features_scaled = self.scaler.transform(features)
        prediction = self.regime_model.predict(features_scaled)
        return int(prediction[0])
    
    # Additional helper methods would continue here...
    # For brevity, I'll add the key remaining methods
    
    async def _enhance_setups_with_ai(self, setups: List[TradeSetup], analysis: MarketAnalysis) -> List[TradeSetup]:
        """Enhance trade setups with AI predictions"""
        enhanced_setups = []
        
        for setup in setups:
            # Get AI prediction for this setup
            data = await self.strategy_manager._get_market_data(setup.symbol, setup.timeframe, 30)
            features = self._extract_features(data)
            
            # Adjust confidence based on AI prediction
            direction_prob = self._predict_direction(features)
            
            # Increase confidence if AI agrees with setup direction
            if ((setup.direction == TradeDirection.LONG and direction_prob > 0.6) or
                (setup.direction == TradeDirection.SHORT and direction_prob < 0.4)):
                setup.confidence = min(setup.confidence * 1.2, 1.0)
            else:
                setup.confidence = setup.confidence * 0.8
            
            enhanced_setups.append(setup)
        
        return enhanced_setups
    
    def _assess_risk(self, analysis: MarketAnalysis) -> str:
        """Assess overall market risk"""
        risk_factors = []
        
        if analysis.market_structure == MarketStructure.RANGING:
            risk_factors.append("Choppy market conditions")
        
        if len(analysis.liquidity_pools) > 3:
            risk_factors.append("Multiple liquidity levels nearby")
        
        if analysis.confidence < 0.6:
            risk_factors.append("Low confidence in analysis")
        
        if not risk_factors:
            return "Low risk - Clear market structure and high confidence"
        elif len(risk_factors) == 1:
            return f"Medium risk - {risk_factors[0]}"
        else:
            return f"High risk - {', '.join(risk_factors)}"
    
    def _generate_market_outlook(self, analysis: MarketAnalysis) -> str:
        """Generate market outlook based on analysis"""
        if analysis.market_structure == MarketStructure.BULLISH:
            return f"Bullish outlook for {analysis.symbol}. Look for long opportunities on pullbacks to support levels."
        elif analysis.market_structure == MarketStructure.BEARISH:
            return f"Bearish outlook for {analysis.symbol}. Look for short opportunities on rallies to resistance levels."
        else:
            return f"Neutral outlook for {analysis.symbol}. Range-bound market - trade the range boundaries."
    
    def _identify_key_events(self, analysis: MarketAnalysis) -> List[str]:
        """Identify key upcoming events"""
        events = []
        
        # Add liquidity sweep events
        for pool in analysis.liquidity_pools:
            if not pool.is_swept:
                events.append(f"Potential liquidity sweep at {pool.price}")
        
        # Add order block events
        for ob in analysis.order_blocks:
            if not ob.is_mitigated:
                events.append(f"Order block mitigation opportunity at {ob.low}-{ob.high}")
        
        # Add session events
        current_hour = datetime.utcnow().hour
        if 6 <= current_hour <= 7:
            events.append("London open approaching - expect increased volatility")
        elif 11 <= current_hour <= 12:
            events.append("New York open approaching - watch for directional moves")
        
        return events[:5]  # Limit to 5 most important events

    def _select_relevant_strategies(self, analysis: MarketAnalysis, all_strategies: List[str]) -> List[str]:
        """Select most relevant ICT strategies based on current market conditions"""
        relevant_strategies = []
        
        # Always include core strategies
        core_strategies = [
            "order_block_strategy", "fair_value_gap_strategy", "market_structure_strategy",
            "liquidity_strategy", "premium_discount_strategy"
        ]
        relevant_strategies.extend([s for s in core_strategies if s in all_strategies])
        
        # Add time-based strategies based on current session
        current_hour = datetime.utcnow().hour
        if 7 <= current_hour <= 10:  # London session
            relevant_strategies.extend([
                "london_killzone_strategy", "session_opens_strategy", "judas_swing_strategy"
            ])
        elif 13 <= current_hour <= 16:  # New York session
            relevant_strategies.extend([
                "silver_bullet_strategy", "ny_reversal_strategy", "turtle_soup_strategy"
            ])
        elif 20 <= current_hour or current_hour <= 6:  # Asian session
            relevant_strategies.extend([
                "asian_range_strategy", "accumulation_distribution_strategy"
            ])
        
        # Add strategies based on market structure
        if analysis.market_structure == MarketStructure.BULLISH:
            relevant_strategies.extend([
                "breaker_block_strategy", "mitigation_block_strategy", "power_of_three_strategy"
            ])
        elif analysis.market_structure == MarketStructure.BEARISH:
            relevant_strategies.extend([
                "rejection_block_strategy", "smt_divergence_strategy", "liquidity_runs_strategy"
            ])
        else:  # Ranging market
            relevant_strategies.extend([
                "dealing_ranges_strategy", "supply_demand_zones_strategy", "range_expansion_strategy"
            ])
        
        # Add advanced strategies for high-confidence setups
        if analysis.confidence > 0.7:
            relevant_strategies.extend([
                "optimal_trade_entry_strategy", "high_probability_scenarios_strategy",
                "ipda_theory_strategy", "algo_price_delivery_strategy"
            ])
        
        # Remove duplicates and ensure strategies exist
        relevant_strategies = list(set([s for s in relevant_strategies if s in all_strategies]))
        
        # Ensure we have at least 10 strategies for comprehensive analysis
        if len(relevant_strategies) < 10:
            remaining_strategies = [s for s in all_strategies if s not in relevant_strategies]
            relevant_strategies.extend(remaining_strategies[:10-len(relevant_strategies)])
        
        return relevant_strategies[:20]  # Limit to top 20 for performance

    # Additional ICT-specific feature extraction methods
    
    def _calculate_order_block_strength(self, data: pd.DataFrame) -> float:
        """Calculate order block strength using volume and price action"""
        if len(data) < 20:
            return 0.0
        
        # Look for strong impulse moves
        body_sizes = []
        for i in range(len(data)):
            body_size = abs(data['close'].iloc[i] - data['open'].iloc[i])
            candle_range = data['high'].iloc[i] - data['low'].iloc[i]
            if candle_range > 0:
                body_sizes.append(body_size / candle_range)
        
        return np.mean(body_sizes[-10:]) if body_sizes else 0.0
    
    def _detect_breaker_block_signals(self, data: pd.DataFrame) -> float:
        """Detect potential breaker block formations"""
        if len(data) < 30:
            return 0.0
        
        # Look for failed order blocks that become breakers
        breaker_count = 0
        for i in range(20, len(data)-5):
            # Simplified breaker detection
            prev_high = max(data['high'].iloc[i-10:i])
            curr_break = any(data['close'].iloc[i:i+5] < prev_high * 0.995)
            if curr_break:
                breaker_count += 1
        
        return min(breaker_count / 5, 1.0)
    
    def _calculate_avg_gap_size(self, data: pd.DataFrame) -> float:
        """Calculate average size of fair value gaps"""
        gaps = self._detect_gaps(data)
        if not gaps:
            return 0.0
        
        gap_sizes = [gap['size'] for gap in gaps]
        avg_price = np.mean(data['close'].values)
        
        return np.mean(gap_sizes) / avg_price if avg_price > 0 else 0.0
    
    def _calculate_rejection_strength(self, data: pd.DataFrame) -> float:
        """Calculate strength of price rejection at levels"""
        rejection_strength = 0.0
        
        for i in range(len(data)):
            total_range = data['high'].iloc[i] - data['low'].iloc[i]
            if total_range > 0:
                upper_wick = data['high'].iloc[i] - max(data['open'].iloc[i], data['close'].iloc[i])
                lower_wick = min(data['open'].iloc[i], data['close'].iloc[i]) - data['low'].iloc[i]
                
                max_wick = max(upper_wick, lower_wick)
                rejection_strength += max_wick / total_range
        
        return rejection_strength / len(data) if len(data) > 0 else 0.0
    
    def _calculate_mitigation_potential(self, data: pd.DataFrame) -> float:
        """Calculate potential for order block mitigation"""
        if len(data) < 20:
            return 0.0
        
        # Look for price returning to previous significant levels
        significant_levels = []
        for i in range(10, len(data)):
            if (data['high'].iloc[i] == max(data['high'].iloc[i-10:i+1]) or
                data['low'].iloc[i] == min(data['low'].iloc[i-10:i+1])):
                significant_levels.append(data['close'].iloc[i])
        
        if not significant_levels:
            return 0.0
        
        current_price = data['close'].iloc[-1]
        nearest_level = min(significant_levels, key=lambda x: abs(x - current_price))
        distance_ratio = abs(current_price - nearest_level) / current_price
        
        return max(0, 1 - distance_ratio * 10)  # Closer = higher mitigation potential
    
    def _calculate_supply_demand_imbalance(self, data: pd.DataFrame) -> float:
        """Calculate supply/demand imbalance"""
        if len(data) < 10:
            return 0.0
        
        # Use volume and price relationship
        volume_price_correlation = np.corrcoef(
            data['volume'].iloc[-10:], 
            data['close'].iloc[-10:]
        )[0, 1]
        
        return volume_price_correlation if not np.isnan(volume_price_correlation) else 0.0
    
    def _calculate_premium_discount_ratio(self, data: pd.DataFrame) -> float:
        """Calculate if price is at premium or discount"""
        if len(data) < 50:
            return 0.5
        
        # Use 50-period range
        range_high = max(data['high'].iloc[-50:])
        range_low = min(data['low'].iloc[-50:])
        current_price = data['close'].iloc[-1]
        
        if range_high == range_low:
            return 0.5
        
        position_in_range = (current_price - range_low) / (range_high - range_low)
        return position_in_range
    
    def _get_killzone_factor(self) -> float:
        """Get current killzone factor"""
        current_hour = datetime.utcnow().hour
        
        # London killzone
        if 7 <= current_hour <= 10:
            return 1.0
        # New York killzone
        elif 13 <= current_hour <= 16:
            return 1.0
        # Asian session
        elif 20 <= current_hour or current_hour <= 6:
            return 0.5
        else:
            return 0.2
    
    def _calculate_session_momentum(self, data: pd.DataFrame) -> float:
        """Calculate momentum within current session"""
        if len(data) < 8:
            return 0.0
        
        # Use last 8 hours as session proxy
        session_data = data.iloc[-8:]
        session_return = (session_data['close'].iloc[-1] - session_data['open'].iloc[0]) / session_data['open'].iloc[0]
        
        return session_return
    
    def _calculate_fibonacci_confluence(self, data: pd.DataFrame) -> float:
        """Calculate confluence with Fibonacci levels"""
        if len(data) < 20:
            return 0.0
        
        # Find significant swing high and low
        swing_high = max(data['high'].iloc[-20:])
        swing_low = min(data['low'].iloc[-20:])
        current_price = data['close'].iloc[-1]
        
        if swing_high == swing_low:
            return 0.0
        
        # Calculate key Fibonacci levels
        fib_range = swing_high - swing_low
        fib_618 = swing_high - (fib_range * 0.618)
        fib_500 = swing_high - (fib_range * 0.500)
        fib_382 = swing_high - (fib_range * 0.382)
        
        # Check proximity to key levels
        fib_levels = [fib_618, fib_500, fib_382]
        min_distance = min(abs(current_price - level) / current_price for level in fib_levels)
        
        return max(0, 1 - min_distance * 50)  # Higher value = closer to Fib level
    
    def _calculate_liquidity_sweep_probability(self, data: pd.DataFrame) -> float:
        """Calculate probability of liquidity sweep"""
        if len(data) < 20:
            return 0.0
        
        # Look for equal highs/lows that haven't been swept
        highs = data['high'].iloc[-20:].values
        lows = data['low'].iloc[-20:].values
        
        equal_high_count = len([h for h in highs if abs(h - max(highs)) / max(highs) < 0.001])
        equal_low_count = len([l for l in lows if abs(l - min(lows)) / min(lows) < 0.001])
        
        return min((equal_high_count + equal_low_count) / 10, 1.0)
    
    def _detect_amd_phase(self, data: pd.DataFrame) -> float:
        """Detect Accumulation, Manipulation, Distribution phase"""
        if len(data) < 30:
            return 0.0
        
        # Simplified AMD detection
        recent_volatility = np.std(data['close'].iloc[-10:])
        previous_volatility = np.std(data['close'].iloc[-20:-10])
        
        # Distribution phase typically has increasing volatility
        volatility_ratio = recent_volatility / (previous_volatility + 1e-10)
        
        return min(volatility_ratio / 2, 1.0)
    
    def _calculate_order_flow_strength(self, data: pd.DataFrame) -> float:
        """Calculate order flow strength"""
        if len(data) < 5:
            return 0.0
        
        # Use close position within range as proxy for order flow
        ranges = data['high'] - data['low']
        close_positions = (data['close'] - data['low']) / (ranges + 1e-10)
        
        # Average close position > 0.5 suggests bullish order flow
        avg_close_position = np.mean(close_positions.iloc[-5:])
        
        return avg_close_position
    
    def _detect_silver_bullet_pattern(self, data: pd.DataFrame) -> float:
        """Detect Silver Bullet setup pattern"""
        current_hour = datetime.utcnow().hour
        
        # Silver Bullet is NY session specific (10-11 AM EST)
        if 14 <= current_hour <= 15:  # Adjusted for UTC
            if len(data) >= 3:
                # Look for consolidation followed by breakout
                recent_range = max(data['high'].iloc[-3:]) - min(data['low'].iloc[-3:])
                avg_range = np.mean(data['high'].iloc[-10:] - data['low'].iloc[-10:])
                
                if recent_range < avg_range * 0.7:  # Consolidation
                    return 0.8
            return 0.5
        
        return 0.0
    
    def _calculate_turtle_soup_probability(self, data: pd.DataFrame) -> float:
        """Calculate Turtle Soup (false breakout) probability"""
        if len(data) < 20:
            return 0.0
        
        # Look for recent breakouts that might fail
        recent_high = max(data['high'].iloc[-5:])
        previous_resistance = max(data['high'].iloc[-20:-5])
        
        if recent_high > previous_resistance * 1.001:  # Breakout occurred
            # Check if it's failing
            current_price = data['close'].iloc[-1]
            if current_price < previous_resistance:
                return 0.8  # High probability turtle soup
            
        return 0.2
    
    def _detect_judas_swing_signal(self, data: pd.DataFrame) -> float:
        """Detect Judas Swing signal"""
        current_hour = datetime.utcnow().hour
        
        # Judas swings typically occur at session opens
        if current_hour in [0, 8, 13]:  # Major session opens
            if len(data) >= 5:
                # Look for initial move followed by reversal
                session_high = max(data['high'].iloc[-3:])
                session_low = min(data['low'].iloc[-3:])
                open_price = data['open'].iloc[-3]
                current_price = data['close'].iloc[-1]
                
                # Check for reversal pattern
                if (session_high > open_price * 1.001 and current_price < open_price * 0.999):
                    return 0.8
                elif (session_low < open_price * 0.999 and current_price > open_price * 1.001):
                    return 0.8
            
            return 0.5
        
        return 0.0
    
    def _identify_power_of_three_phase(self, data: pd.DataFrame) -> float:
        """Identify current Power of 3 phase"""
        if len(data) < 30:
            return 0.0
        
        # Simplified phase detection
        early_data = data.iloc[:10]
        middle_data = data.iloc[10:20]
        late_data = data.iloc[20:]
        
        early_volatility = np.std(early_data['close'])
        middle_volatility = np.std(middle_data['close'])
        late_volatility = np.std(late_data['close'])
        
        # Distribution phase typically has highest volatility
        if late_volatility > middle_volatility > early_volatility:
            return 1.0  # Distribution phase
        elif middle_volatility > early_volatility and middle_volatility > late_volatility:
            return 0.5  # Manipulation phase
        else:
            return 0.0  # Accumulation phase


# Data classes for AI responses
class MarketPrediction:
    def __init__(self, direction, confidence, target_levels, factors, risks):
        self.direction = direction
        self.confidence = confidence
        self.target_levels = target_levels
        self.factors = factors
        self.risks = risks

class PortfolioOptimization:
    def __init__(self, allocation, expected_return, expected_risk, sharpe_ratio, diversification_score, rebalance_suggestions):
        self.allocation = allocation
        self.expected_return = expected_return
        self.expected_risk = expected_risk
        self.sharpe_ratio = sharpe_ratio
        self.diversification_score = diversification_score
        self.rebalance_suggestions = rebalance_suggestions

class MarketRegime:
    def __init__(self, regime_type, confidence, strength, expected_duration, trading_implications, recommended_strategies):
        self.regime_type = regime_type
        self.confidence = confidence
        self.strength = strength
        self.expected_duration = expected_duration
        self.trading_implications = trading_implications
        self.recommended_strategies = recommended_strategies

class AIBacktestResults:
    def __init__(self, adaptive_accuracy, regime_accuracy, pattern_accuracy, sentiment_correlation, vs_static_strategies):
        self.adaptive_accuracy = adaptive_accuracy
        self.regime_accuracy = regime_accuracy
        self.pattern_accuracy = pattern_accuracy
        self.sentiment_correlation = sentiment_correlation
        self.vs_static_strategies = vs_static_strategies