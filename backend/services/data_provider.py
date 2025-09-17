"""
Real-time data provider for market data
Supports multiple data sources and custom symbol input
"""
import aiohttp
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os
import json

class RealDataProvider:
    """Real-time data provider for financial markets"""
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', 'demo')
        self.session = None
        
    async def _get_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def get_real_time_data(self, symbol: str, interval: str = '1min') -> Dict[str, Any]:
        """Get real-time data for a symbol"""
        try:
            # Try multiple data sources in order of preference
            data = await self._get_yahoo_finance_data(symbol)
            if not data:
                data = await self._get_alpha_vantage_data(symbol, interval)
            if not data:
                data = await self._get_fallback_data(symbol)
                
            return data
        except Exception as e:
            print(f"Error fetching real data for {symbol}: {e}")
            return await self._get_fallback_data(symbol)
    
    async def _get_yahoo_finance_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get data from Yahoo Finance (free alternative)"""
        try:
            session = await self._get_session()
            
            # Convert symbol format for Yahoo Finance
            yahoo_symbol = self._format_symbol_for_yahoo(symbol)
            
            # Yahoo Finance API endpoint (unofficial but free)
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
            params = {
                'interval': '1m',
                'range': '1d',
                'includePrePost': 'false'
            }
            
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_yahoo_response(data, symbol)
                    
        except Exception as e:
            print(f"Yahoo Finance API error for {symbol}: {e}")
            
        return None
    
    async def _get_alpha_vantage_data(self, symbol: str, interval: str = '1min') -> Optional[Dict[str, Any]]:
        """Get data from Alpha Vantage API"""
        try:
            session = await self._get_session()
            
            # Convert symbol format for different markets
            av_symbol = self._format_symbol_for_alpha_vantage(symbol)
            
            url = f"https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_INTRADAY',
                'symbol': av_symbol,
                'interval': interval,
                'apikey': self.alpha_vantage_key,
                'outputsize': 'compact'
            }
            
            async with session.get(url, params=params, timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_alpha_vantage_response(data, symbol)
                    
        except Exception as e:
            print(f"Alpha Vantage API error: {e}")
            
        return None
    
    async def _get_fallback_data(self, symbol: str) -> Dict[str, Any]:
        """Enhanced fallback using yfinance for real market data"""
        try:
            import yfinance as yf
            
            # Format symbol for yfinance
            formatted_symbol = self._format_symbol_for_yfinance(symbol)
            
            ticker = yf.Ticker(formatted_symbol)
            hist = ticker.history(period="5d", interval="1m")
            
            if hist.empty:
                # Try alternative formatting
                alt_symbol = self._get_alternative_symbol_format(symbol)
                if alt_symbol != formatted_symbol:
                    ticker = yf.Ticker(alt_symbol)
                    hist = ticker.history(period="5d", interval="1m")
            
            if hist.empty:
                raise Exception(f"No data available for symbol {symbol}")
            
            current_time = datetime.utcnow()
            ohlcv_data = []
            
            for timestamp, row in hist.iterrows():
                ohlcv_data.append({
                    'timestamp': timestamp.isoformat(),
                    'open': round(float(row['Open']), self._get_price_decimals(symbol)),
                    'high': round(float(row['High']), self._get_price_decimals(symbol)),
                    'low': round(float(row['Low']), self._get_price_decimals(symbol)),
                    'close': round(float(row['Close']), self._get_price_decimals(symbol)),
                    'volume': int(row['Volume'])
                })
            
            if not ohlcv_data:
                raise Exception(f"No valid data points for {symbol}")
            
            return {
                'symbol': symbol,
                'data': ohlcv_data,
                'current_price': ohlcv_data[-1]['close'],
                'price_change_24h': ((ohlcv_data[-1]['close'] - ohlcv_data[0]['close']) / ohlcv_data[0]['close']) * 100,
                'data_source': 'yfinance',
                'timestamp': current_time.isoformat(),
                'disclaimer': 'Real market data via Yahoo Finance'
            }
            
        except Exception as e:
            print(f"Enhanced yfinance failed for {symbol}: {e}")
            raise Exception(f"Unable to fetch real market data for {symbol}: {str(e)}")
    
    def _format_symbol_for_yfinance(self, symbol: str) -> str:
        """Format symbol for Yahoo Finance"""
        # Forex pairs
        forex_map = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'USDCHF': 'USDCHF=X',
            'NZDUSD': 'NZDUSD=X'
        }
        
        if symbol in forex_map:
            return forex_map[symbol]
        
        # Stocks - use as is
        return symbol
    
    def _get_alternative_symbol_format(self, symbol: str) -> str:
        """Get alternative symbol formats"""
        if symbol.endswith('=X'):
            return symbol[:-2]  # Remove =X
        elif symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']:
            return f"{symbol}=X"  # Add =X
        return symbol
    
    def _format_symbol_for_alpha_vantage(self, symbol: str) -> str:
        """Format symbol for Alpha Vantage API"""
        if '.NS' in symbol:
            # Indian stocks - remove .NS suffix for Alpha Vantage
            return symbol.replace('.NS', '.BSE')
        elif symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']:
            # Forex pairs
            return symbol
        else:
            # US stocks
            return symbol
    
    def _format_symbol_for_yahoo(self, symbol: str) -> str:
        """Format symbol for Yahoo Finance"""
        return self._format_symbol_for_yfinance(symbol)
    
    def _get_realistic_base_price(self, symbol: str) -> float:
        """Get realistic base prices for different symbols"""
        base_prices = {
            'EURUSD': 1.0800,
            'GBPUSD': 1.2650,
            'USDJPY': 148.50,
            'AUDUSD': 0.6850,
            'USDCAD': 1.3520,
            'IRFC.NS': 147.50,      # Current IRFC price (₹147.50)
            'RELIANCE.NS': 2850.00,
            'TCS.NS': 3200.00,
            'INFY.NS': 1450.00,
            'HDFC.NS': 1680.00,
            'AAPL': 175.00,
            'TSLA': 240.00,
            'MSFT': 380.00,
            'GOOGL': 140.00,
        }
        return base_prices.get(symbol, 100.00)
    
    def _get_price_decimals(self, symbol: str) -> int:
        """Get appropriate decimal places for symbol"""
        if 'JPY' in symbol:
            return 3
        elif '.NS' in symbol or symbol in ['AAPL', 'TSLA', 'MSFT', 'GOOGL']:
            return 2
        else:
            return 5  # Forex pairs
    
    def _parse_alpha_vantage_response(self, data: Dict, symbol: str) -> Optional[Dict[str, Any]]:
        """Parse Alpha Vantage API response"""
        try:
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
                    
            if not time_series_key:
                return None
                
            time_series = data[time_series_key]
            ohlcv_data = []
            
            for timestamp, values in time_series.items():
                ohlcv_data.append({
                    'timestamp': timestamp,
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            # Sort by timestamp
            ohlcv_data.sort(key=lambda x: x['timestamp'])
            
            if ohlcv_data:
                latest = ohlcv_data[-1]
                oldest = ohlcv_data[0]
                price_change = ((latest['close'] - oldest['close']) / oldest['close']) * 100
                
                return {
                    'symbol': symbol,
                    'data': ohlcv_data,
                    'current_price': latest['close'],
                    'price_change_24h': price_change,
                    'data_source': 'alpha_vantage',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            print(f"Error parsing Alpha Vantage response: {e}")
            
        return None
    
    def _parse_yahoo_response(self, data: Dict, symbol: str) -> Optional[Dict[str, Any]]:
        """Parse Yahoo Finance API response"""
        try:
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            quotes = result['indicators']['quote'][0]
            
            ohlcv_data = []
            for i, timestamp in enumerate(timestamps):
                if (quotes['open'][i] is not None and 
                    quotes['high'][i] is not None and 
                    quotes['low'][i] is not None and 
                    quotes['close'][i] is not None):
                    
                    ohlcv_data.append({
                        'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
                        'open': float(quotes['open'][i]),
                        'high': float(quotes['high'][i]),
                        'low': float(quotes['low'][i]),
                        'close': float(quotes['close'][i]),
                        'volume': int(quotes['volume'][i]) if quotes['volume'][i] else 0
                    })
            
            if ohlcv_data:
                latest = ohlcv_data[-1]
                oldest = ohlcv_data[0]
                price_change = ((latest['close'] - oldest['close']) / oldest['close']) * 100
                
                return {
                    'symbol': symbol,
                    'data': ohlcv_data,
                    'current_price': latest['close'],
                    'price_change_24h': price_change,
                    'data_source': 'yahoo_finance',
                    'timestamp': datetime.utcnow().isoformat()
                }
                
        except Exception as e:
            print(f"Error parsing Yahoo Finance response: {e}")
            
        return None
    
    async def validate_symbol(self, symbol: str) -> bool:
        """Validate if a symbol exists and can be traded"""
        try:
            data = await self.get_real_time_data(symbol)
            return data is not None and len(data.get('data', [])) > 0
        except:
            return False
    
    async def search_symbols(self, query: str) -> List[Dict[str, str]]:
        """Search for symbols matching the query"""
        # This would ideally connect to a symbol search API
        # For now, return some common symbols that match the query
        common_symbols = [
            {'symbol': 'IRFC.NS', 'name': 'Indian Railway Finance Corporation', 'exchange': 'NSE'},
            {'symbol': 'RELIANCE.NS', 'name': 'Reliance Industries Limited', 'exchange': 'NSE'},
            {'symbol': 'TCS.NS', 'name': 'Tata Consultancy Services', 'exchange': 'NSE'},
            {'symbol': 'INFY.NS', 'name': 'Infosys Limited', 'exchange': 'NSE'},
            {'symbol': 'HDFC.NS', 'name': 'HDFC Bank Limited', 'exchange': 'NSE'},
            {'symbol': 'EURUSD', 'name': 'EUR/USD', 'exchange': 'FOREX'},
            {'symbol': 'GBPUSD', 'name': 'GBP/USD', 'exchange': 'FOREX'},
            {'symbol': 'USDJPY', 'name': 'USD/JPY', 'exchange': 'FOREX'},
            {'symbol': 'AAPL', 'name': 'Apple Inc.', 'exchange': 'NASDAQ'},
            {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'exchange': 'NASDAQ'},
        ]
        
        query_lower = query.lower()
        return [s for s in common_symbols 
                if query_lower in s['symbol'].lower() or query_lower in s['name'].lower()]
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()

# Global data provider instance
data_provider = RealDataProvider()