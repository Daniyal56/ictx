import asyncio
import aiohttp
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import re

class SentimentAnalyzer:
    """Analyze market sentiment from multiple sources"""
    
    def __init__(self):
        self.sources = {
            'news': self._analyze_news_sentiment,
            'social': self._analyze_social_sentiment,
            'technical': self._analyze_technical_sentiment,
            'options_flow': self._analyze_options_flow_sentiment
        }
    
    async def analyze_market_sentiment(self, symbol: str) -> str:
        """Analyze sentiment for a specific symbol"""
        sentiments = []
        
        # Technical sentiment (based on price action)
        technical_sentiment = await self._analyze_technical_sentiment(symbol)
        sentiments.append(technical_sentiment)
        
        # News sentiment (mock implementation)
        news_sentiment = await self._analyze_news_sentiment(symbol)
        sentiments.append(news_sentiment)
        
        # Average sentiment
        avg_sentiment = sum(sentiments) / len(sentiments)
        
        if avg_sentiment > 0.6:
            return "bullish"
        elif avg_sentiment < 0.4:
            return "bearish"
        else:
            return "neutral"
    
    async def comprehensive_analysis(
        self, 
        symbol: str, 
        sources: List[str]
    ) -> Dict[str, Any]:
        """Comprehensive sentiment analysis from multiple sources"""
        
        sentiment_scores = {}
        
        for source in sources:
            if source in self.sources:
                try:
                    score = await self.sources[source](symbol)
                    sentiment_scores[source] = score
                except Exception as e:
                    sentiment_scores[source] = 0.5  # Neutral if analysis fails
        
        # Calculate overall sentiment
        if sentiment_scores:
            overall_sentiment = sum(sentiment_scores.values()) / len(sentiment_scores)
        else:
            overall_sentiment = 0.5
        
        # Calculate confidence based on agreement between sources
        if len(sentiment_scores) > 1:
            sentiment_values = list(sentiment_scores.values())
            variance = sum((x - overall_sentiment) ** 2 for x in sentiment_values) / len(sentiment_values)
            confidence = max(0, 1 - variance * 2)  # Higher variance = lower confidence
        else:
            confidence = 0.5
        
        # Identify key factors
        key_factors = self._identify_key_sentiment_factors(sentiment_scores, symbol)
        
        return {
            "overall": self._score_to_sentiment(overall_sentiment),
            "breakdown": {
                source: {
                    "score": score,
                    "sentiment": self._score_to_sentiment(score)
                }
                for source, score in sentiment_scores.items()
            },
            "confidence": confidence,
            "factors": key_factors
        }
    
    async def _analyze_technical_sentiment(self, symbol: str) -> float:
        """Analyze technical sentiment based on price action"""
        # Mock technical analysis
        # In real implementation, this would analyze:
        # - RSI levels
        # - Moving average positioning
        # - Support/resistance levels
        # - Volume patterns
        # - Momentum indicators
        
        # Simulate technical indicators
        import random
        random.seed(hash(symbol) % 1000)
        
        # Mock RSI
        rsi = random.uniform(20, 80)
        rsi_sentiment = (rsi - 50) / 50  # Convert to -1 to 1 scale
        
        # Mock price vs moving averages
        price_ma_sentiment = random.uniform(-0.5, 0.5)
        
        # Mock momentum
        momentum_sentiment = random.uniform(-0.3, 0.3)
        
        # Combine technical factors
        technical_score = (rsi_sentiment + price_ma_sentiment + momentum_sentiment) / 3
        
        # Convert to 0-1 scale
        return (technical_score + 1) / 2
    
    async def _analyze_news_sentiment(self, symbol: str) -> float:
        """Analyze news sentiment (mock implementation)"""
        # Mock news sentiment analysis
        # In real implementation, this would:
        # - Fetch recent news articles about the symbol
        # - Use NLP to analyze sentiment
        # - Weight by source credibility and recency
        
        # Simulate news sentiment based on symbol
        news_keywords = {
            'EURUSD': ['ECB', 'inflation', 'employment', 'GDP'],
            'GBPUSD': ['BOE', 'Brexit', 'inflation', 'employment'],
            'USDJPY': ['Fed', 'BOJ', 'inflation', 'yields'],
            'AAPL': ['earnings', 'iPhone', 'innovation', 'revenue'],
            'TSLA': ['electric', 'autonomous', 'production', 'delivery']
        }
        
        # Mock sentiment based on recent "news"
        base_sentiment = 0.5
        
        if symbol in news_keywords:
            # Simulate positive/negative news impact
            import random
            random.seed(hash(symbol + "news") % 1000)
            
            news_impact = random.uniform(-0.3, 0.3)
            base_sentiment += news_impact
        
        return max(0, min(1, base_sentiment))
    
    async def _analyze_social_sentiment(self, symbol: str) -> float:
        """Analyze social media sentiment (mock implementation)"""
        # Mock social sentiment analysis
        # In real implementation, this would:
        # - Scrape Twitter, Reddit, Discord, Telegram
        # - Analyze mentions and sentiment
        # - Weight by follower count and engagement
        
        import random
        random.seed(hash(symbol + "social") % 1000)
        
        # Simulate social sentiment
        social_buzz = random.uniform(0, 1)
        sentiment_polarity = random.uniform(-0.4, 0.4)
        
        # Higher buzz amplifies sentiment
        social_sentiment = 0.5 + (sentiment_polarity * social_buzz)
        
        return max(0, min(1, social_sentiment))
    
    async def _analyze_options_flow_sentiment(self, symbol: str) -> float:
        """Analyze options flow sentiment (mock implementation)"""
        # Mock options flow analysis
        # In real implementation, this would:
        # - Analyze unusual options activity
        # - Look at put/call ratios
        # - Analyze large block trades
        # - Consider implied volatility changes
        
        import random
        random.seed(hash(symbol + "options") % 1000)
        
        # Simulate put/call ratio
        put_call_ratio = random.uniform(0.5, 2.0)
        
        # Lower put/call ratio = more bullish
        if put_call_ratio < 0.8:
            options_sentiment = 0.7  # Bullish
        elif put_call_ratio > 1.2:
            options_sentiment = 0.3  # Bearish
        else:
            options_sentiment = 0.5  # Neutral
        
        # Add noise for unusual activity
        unusual_activity = random.uniform(-0.2, 0.2)
        options_sentiment += unusual_activity
        
        return max(0, min(1, options_sentiment))
    
    def _score_to_sentiment(self, score: float) -> str:
        """Convert numerical score to sentiment label"""
        if score > 0.6:
            return "bullish"
        elif score < 0.4:
            return "bearish"
        else:
            return "neutral"
    
    def _identify_key_sentiment_factors(
        self, 
        sentiment_scores: Dict[str, float], 
        symbol: str
    ) -> List[str]:
        """Identify key factors driving sentiment"""
        factors = []
        
        # Technical factors
        if 'technical' in sentiment_scores:
            technical_score = sentiment_scores['technical']
            if technical_score > 0.7:
                factors.append("Strong technical indicators showing bullish momentum")
            elif technical_score < 0.3:
                factors.append("Weak technical indicators showing bearish pressure")
            elif 0.4 <= technical_score <= 0.6:
                factors.append("Technical indicators showing neutral/mixed signals")
        
        # News factors
        if 'news' in sentiment_scores:
            news_score = sentiment_scores['news']
            if news_score > 0.6:
                factors.append("Positive news flow and fundamental developments")
            elif news_score < 0.4:
                factors.append("Negative news sentiment impacting outlook")
        
        # Social factors
        if 'social' in sentiment_scores:
            social_score = sentiment_scores['social']
            if social_score > 0.6:
                factors.append("Positive social media sentiment and retail interest")
            elif social_score < 0.4:
                factors.append("Negative social sentiment and declining retail interest")
        
        # Options flow factors
        if 'options_flow' in sentiment_scores:
            options_score = sentiment_scores['options_flow']
            if options_score > 0.6:
                factors.append("Bullish options flow with low put/call ratio")
            elif options_score < 0.4:
                factors.append("Bearish options positioning with high put activity")
        
        # Cross-source agreement
        scores = list(sentiment_scores.values())
        if len(scores) > 1:
            variance = sum((x - sum(scores)/len(scores)) ** 2 for x in scores) / len(scores)
            if variance < 0.1:
                factors.append("Strong agreement across all sentiment sources")
            elif variance > 0.3:
                factors.append("Mixed signals with conflicting sentiment sources")
        
        # Market-specific factors
        if symbol.endswith('USD=X') or 'USD' in symbol:
            factors.append("USD strength/weakness affecting sentiment")
        
        if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:
            factors.append("Large-cap tech sentiment influencing overall market mood")
        
        return factors[:5]  # Return top 5 factors
    
    async def get_sentiment_history(
        self, 
        symbol: str, 
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get historical sentiment data (mock implementation)"""
        history = []
        
        for i in range(days):
            date = datetime.utcnow() - timedelta(days=days-i)
            
            # Mock historical sentiment
            import random
            random.seed(hash(symbol + str(date.date())) % 1000)
            
            sentiment_score = random.uniform(0.2, 0.8)
            
            history.append({
                'date': date.date(),
                'sentiment_score': sentiment_score,
                'sentiment_label': self._score_to_sentiment(sentiment_score),
                'confidence': random.uniform(0.5, 0.9),
                'volume_mentions': random.randint(50, 500)
            })
        
        return history
    
    async def detect_sentiment_anomalies(
        self, 
        symbol: str
    ) -> List[Dict[str, Any]]:
        """Detect unusual sentiment patterns"""
        anomalies = []
        
        # Get recent sentiment history
        history = await self.get_sentiment_history(symbol, 7)
        
        if len(history) < 3:
            return anomalies
        
        # Calculate moving average
        recent_scores = [h['sentiment_score'] for h in history[-3:]]
        avg_sentiment = sum(recent_scores) / len(recent_scores)
        
        # Check for sudden sentiment shifts
        if len(history) >= 2:
            prev_sentiment = history[-2]['sentiment_score']
            current_sentiment = history[-1]['sentiment_score']
            
            sentiment_change = abs(current_sentiment - prev_sentiment)
            
            if sentiment_change > 0.3:  # Significant shift
                anomalies.append({
                    'type': 'sudden_shift',
                    'description': f"Sudden sentiment shift from {self._score_to_sentiment(prev_sentiment)} to {self._score_to_sentiment(current_sentiment)}",
                    'magnitude': sentiment_change,
                    'date': history[-1]['date'],
                    'severity': 'high' if sentiment_change > 0.5 else 'medium'
                })
        
        # Check for extreme sentiment
        if history[-1]['sentiment_score'] > 0.85:
            anomalies.append({
                'type': 'extreme_bullish',
                'description': "Extremely bullish sentiment detected - potential contrarian signal",
                'magnitude': history[-1]['sentiment_score'],
                'date': history[-1]['date'],
                'severity': 'medium'
            })
        elif history[-1]['sentiment_score'] < 0.15:
            anomalies.append({
                'type': 'extreme_bearish',
                'description': "Extremely bearish sentiment detected - potential contrarian signal",
                'magnitude': 1 - history[-1]['sentiment_score'],
                'date': history[-1]['date'],
                'severity': 'medium'
            })
        
        # Check for sentiment divergence with price (would need price data)
        # This is a placeholder for more sophisticated analysis
        
        return anomalies
    
    async def correlate_sentiment_with_performance(
        self, 
        symbol: str, 
        days: int = 30
    ) -> Dict[str, Any]:
        """Correlate sentiment with price performance"""
        
        # Get sentiment history
        sentiment_history = await self.get_sentiment_history(symbol, days)
        
        # Mock price performance data
        import random
        price_changes = []
        
        for i, sentiment_data in enumerate(sentiment_history):
            random.seed(hash(symbol + str(sentiment_data['date'])) % 1000)
            
            # Loosely correlate price change with sentiment
            base_change = (sentiment_data['sentiment_score'] - 0.5) * 0.02  # Max 1% bias
            noise = random.uniform(-0.015, 0.015)  # Random noise
            price_change = base_change + noise
            
            price_changes.append(price_change)
        
        # Calculate correlation
        if len(sentiment_history) > 5:
            sentiment_scores = [h['sentiment_score'] for h in sentiment_history]
            
            # Simple correlation calculation
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
            avg_price_change = sum(price_changes) / len(price_changes)
            
            numerator = sum((s - avg_sentiment) * (p - avg_price_change) 
                          for s, p in zip(sentiment_scores, price_changes))
            
            sentiment_variance = sum((s - avg_sentiment) ** 2 for s in sentiment_scores)
            price_variance = sum((p - avg_price_change) ** 2 for p in price_changes)
            
            if sentiment_variance > 0 and price_variance > 0:
                correlation = numerator / (sentiment_variance * price_variance) ** 0.5
            else:
                correlation = 0
        else:
            correlation = 0
        
        return {
            'correlation_coefficient': correlation,
            'correlation_strength': self._interpret_correlation(correlation),
            'sentiment_predictive_power': abs(correlation),
            'analysis_period_days': days,
            'data_points': len(sentiment_history),
            'average_sentiment': sum(h['sentiment_score'] for h in sentiment_history) / len(sentiment_history),
            'sentiment_volatility': self._calculate_sentiment_volatility(sentiment_history)
        }
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient"""
        abs_corr = abs(correlation)
        
        if abs_corr > 0.7:
            return "strong"
        elif abs_corr > 0.4:
            return "moderate" 
        elif abs_corr > 0.2:
            return "weak"
        else:
            return "negligible"
    
    def _calculate_sentiment_volatility(
        self, 
        sentiment_history: List[Dict[str, Any]]
    ) -> float:
        """Calculate sentiment volatility"""
        if len(sentiment_history) < 2:
            return 0
        
        scores = [h['sentiment_score'] for h in sentiment_history]
        avg_score = sum(scores) / len(scores)
        variance = sum((score - avg_score) ** 2 for score in scores) / len(scores)
        
        return variance ** 0.5