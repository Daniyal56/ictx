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
        """Generate AI-powered trading recommendation"""
        
        # Get current trade setups
        setups = await self.strategy_manager.get_trade_setups(
            analysis.symbol, 
            analysis.timeframe, 
            ["order_block_strategy", "fair_value_gap_strategy", "silver_bullet_strategy"]
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
        """Extract ICT-specific features"""
        features = []
        
        # Market structure features
        highs = data['high'].values
        lows = data['low'].values
        
        # Higher highs / Lower lows pattern
        recent_high = max(highs[-10:])
        previous_high = max(highs[-20:-10])
        higher_high = 1 if recent_high > previous_high else 0
        
        recent_low = min(lows[-10:])
        previous_low = min(lows[-20:-10])
        higher_low = 1 if recent_low > previous_low else 0
        
        features.extend([higher_high, higher_low])
        
        # Liquidity features (simplified)
        equal_highs = self._detect_equal_levels(highs[-20:])
        equal_lows = self._detect_equal_levels(lows[-20:])
        
        features.extend([equal_highs, equal_lows])
        
        # Gap features
        gaps = self._detect_gaps(data)
        features.append(len(gaps))
        
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