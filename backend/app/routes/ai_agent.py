from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from datetime import datetime
from app.models import MarketAnalysis, AIRecommendation, TradeSetup, TimeFrame
from ai.agent import ICTAIAgent
# Temporarily disable AI imports for compatibility
# from ai.pattern_recognition import PatternRecognizer
# from ai.sentiment_analyzer import SentimentAnalyzer

router = APIRouter()
ai_agent = ICTAIAgent()
# Temporarily disable AI functionality
# pattern_recognizer = PatternRecognizer()
# sentiment_analyzer = SentimentAnalyzer()

@router.post("/analyze", response_model=AIRecommendation)
async def ai_market_analysis(
    symbol: str,
    timeframe: TimeFrame = TimeFrame.H4,
    include_sentiment: bool = True
):
    """Get comprehensive AI-powered market analysis"""
    try:
        # Get market analysis
        analysis = await ai_agent.analyze_market(symbol, timeframe)
        
        # Add sentiment analysis if requested
        if include_sentiment:
            sentiment = await sentiment_analyzer.analyze_market_sentiment(symbol)
            analysis.sentiment = sentiment
        
        # Generate AI recommendations
        recommendation = await ai_agent.generate_recommendation(analysis)
        
        return recommendation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI analysis failed: {str(e)}")

@router.post("/patterns")
async def detect_patterns(
    symbol: str,
    timeframe: TimeFrame = TimeFrame.H1,
    lookback_periods: int = 100
):
    """Detect ICT patterns using AI pattern recognition"""
    try:
        patterns = await pattern_recognizer.detect_patterns(
            symbol=symbol,
            timeframe=timeframe,
            lookback_periods=lookback_periods
        )
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "patterns_detected": patterns,
            "pattern_count": len(patterns),
            "analysis_timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern detection failed: {str(e)}")

@router.get("/capabilities")
async def get_ai_capabilities():
    """Get AI agent capabilities and features"""
    return {
        "core_capabilities": [
            "Market Structure Analysis",
            "Pattern Recognition",
            "Sentiment Analysis", 
            "Trade Setup Generation",
            "Risk Assessment",
            "Multi-timeframe Analysis",
            "Correlation Analysis",
            "News Impact Assessment"
        ],
        "ict_expertise": [
            "Order Block Detection",
            "Fair Value Gap Identification",
            "Liquidity Pool Recognition",
            "Market Maker Model Analysis",
            "Session-based Analysis",
            "SMT Divergence Detection",
            "Killzone Optimization",
            "Power of 3 Recognition"
        ],
        "ai_features": [
            "Machine Learning Pattern Recognition",
            "Ensemble Strategy Recommendations",
            "Dynamic Risk Adjustment",
            "Market Regime Detection",
            "Adaptive Strategy Selection",
            "Performance Prediction",
            "Anomaly Detection",
            "Real-time Learning"
        ]
    }

@router.post("/sentiment")
async def analyze_sentiment(
    symbol: str,
    sources: List[str] = None
):
    """Analyze market sentiment from multiple sources"""
    try:
        if sources is None:
            sources = ["news", "social", "technical", "options_flow"]
        
        sentiment_analysis = await sentiment_analyzer.comprehensive_analysis(symbol, sources)
        
        return {
            "symbol": symbol,
            "overall_sentiment": sentiment_analysis["overall"],
            "sentiment_breakdown": sentiment_analysis["breakdown"],
            "confidence": sentiment_analysis["confidence"],
            "key_factors": sentiment_analysis["factors"],
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@router.post("/prediction")
async def predict_market_move(
    symbol: str,
    timeframe: TimeFrame,
    prediction_horizon: int = 24  # hours
):
    """Predict market movement using AI models"""
    try:
        prediction = await ai_agent.predict_market_movement(
            symbol=symbol,
            timeframe=timeframe,
            horizon_hours=prediction_horizon
        )
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "prediction_horizon_hours": prediction_horizon,
            "predicted_direction": prediction.direction,
            "confidence": prediction.confidence,
            "target_levels": prediction.target_levels,
            "key_factors": prediction.factors,
            "risk_factors": prediction.risks,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market prediction failed: {str(e)}")

@router.post("/optimize-portfolio")
async def optimize_portfolio(
    symbols: List[str],
    risk_tolerance: float = 0.02,
    max_positions: int = 5
):
    """Optimize portfolio allocation using AI"""
    try:
        optimization = await ai_agent.optimize_portfolio(
            symbols=symbols,
            risk_tolerance=risk_tolerance,
            max_positions=max_positions
        )
        
        return {
            "recommended_allocation": optimization.allocation,
            "expected_return": optimization.expected_return,
            "expected_risk": optimization.expected_risk,
            "sharpe_ratio": optimization.sharpe_ratio,
            "diversification_score": optimization.diversification_score,
            "rebalance_suggestions": optimization.rebalance_suggestions,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Portfolio optimization failed: {str(e)}")

@router.get("/market-regime")
async def detect_market_regime(
    symbol: str,
    lookback_days: int = 30
):
    """Detect current market regime (trending, ranging, volatile)"""
    try:
        regime = await ai_agent.detect_market_regime(symbol, lookback_days)
        
        return {
            "symbol": symbol,
            "current_regime": regime.regime_type,
            "confidence": regime.confidence,
            "regime_strength": regime.strength,
            "expected_duration": regime.expected_duration,
            "trading_implications": regime.trading_implications,
            "recommended_strategies": regime.recommended_strategies,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regime detection failed: {str(e)}")

@router.post("/backtest-ai")
async def backtest_ai_strategies(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 10000
):
    """Backtest AI-powered trading strategies"""
    try:
        results = await ai_agent.backtest_ai_strategies(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital
        )
        
        return {
            "backtest_results": results,
            "ai_performance_metrics": {
                "adaptive_accuracy": results.adaptive_accuracy,
                "regime_detection_accuracy": results.regime_accuracy,
                "pattern_recognition_accuracy": results.pattern_accuracy,
                "sentiment_correlation": results.sentiment_correlation
            },
            "comparison_vs_static": results.vs_static_strategies,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI backtest failed: {str(e)}")