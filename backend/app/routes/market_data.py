from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta
import yfinance as yf
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import CandleData, TimeFrame

router = APIRouter()

@router.get("/candles/{symbol}")
async def get_market_data(
    symbol: str,
    timeframe: TimeFrame = TimeFrame.H1,
    limit: int = Query(default=100, le=1000),
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """Get historical market data for a symbol"""
    try:
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Map timeframe to yfinance interval
        interval_map = {
            TimeFrame.M1: "1m",
            TimeFrame.M5: "5m", 
            TimeFrame.M15: "15m",
            TimeFrame.M30: "30m",
            TimeFrame.H1: "1h",
            TimeFrame.H4: "4h",
            TimeFrame.D1: "1d",
            TimeFrame.W1: "1wk"
        }
        
        # Fetch data from yfinance
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            start=start_date,
            end=end_date,
            interval=interval_map[timeframe]
        )
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        # Convert to our format
        candles = []
        for timestamp, row in data.iterrows():
            candle = CandleData(
                timestamp=timestamp,
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=float(row['Volume']),
                symbol=symbol,
                timeframe=timeframe
            )
            candles.append(candle)
        
        # Limit results
        candles = candles[-limit:]
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "count": len(candles),
            "start_date": candles[0].timestamp if candles else None,
            "end_date": candles[-1].timestamp if candles else None,
            "data": candles
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {str(e)}")

@router.get("/symbols")
async def get_available_symbols():
    """Get list of available trading symbols"""
    # Major forex pairs
    forex_pairs = [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
        "USDCHF=X", "NZDUSD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X"
    ]
    
    # Major indices
    indices = [
        "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX",
        "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI"
    ]
    
    # Major stocks
    stocks = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
        "META", "NVDA", "JPM", "JNJ", "V"
    ]
    
    # Commodities
    commodities = [
        "GC=F", "SI=F", "CL=F", "NG=F", "HG=F"
    ]
    
    return {
        "categories": {
            "forex": {
                "name": "Forex Pairs",
                "symbols": forex_pairs,
                "description": "Major currency pairs"
            },
            "indices": {
                "name": "Stock Indices", 
                "symbols": indices,
                "description": "Major stock market indices"
            },
            "stocks": {
                "name": "Stocks",
                "symbols": stocks,
                "description": "Major individual stocks"
            },
            "commodities": {
                "name": "Commodities",
                "symbols": commodities,
                "description": "Major commodity futures"
            }
        },
        "total_symbols": len(forex_pairs) + len(indices) + len(stocks) + len(commodities)
    }

@router.get("/quote/{symbol}")
async def get_real_time_quote(symbol: str):
    """Get real-time quote for a symbol"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        # Get recent data
        data = ticker.history(period="1d", interval="1m")
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {symbol}")
        
        latest = data.iloc[-1]
        
        return {
            "symbol": symbol,
            "name": info.get("longName", symbol),
            "price": float(latest['Close']),
            "change": float(latest['Close'] - data.iloc[-2]['Close']) if len(data) > 1 else 0,
            "change_percent": float((latest['Close'] - data.iloc[-2]['Close']) / data.iloc[-2]['Close'] * 100) if len(data) > 1 else 0,
            "volume": float(latest['Volume']),
            "high": float(data['High'].max()),
            "low": float(data['Low'].min()),
            "open": float(data.iloc[0]['Open']),
            "timestamp": latest.name,
            "market_cap": info.get("marketCap"),
            "sector": info.get("sector"),
            "industry": info.get("industry")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get quote: {str(e)}")

@router.get("/market-hours")
async def get_market_hours():
    """Get current market session information"""
    current_time = datetime.utcnow()
    
    # Define market sessions (UTC times)
    sessions = {
        "sydney": {"open": "21:00", "close": "06:00", "name": "Sydney"},
        "tokyo": {"open": "23:00", "close": "08:00", "name": "Tokyo"},
        "london": {"open": "07:00", "close": "16:00", "name": "London"},
        "new_york": {"open": "12:30", "close": "21:00", "name": "New York"}
    }
    
    current_hour = current_time.hour
    current_sessions = []
    
    for session_id, session in sessions.items():
        open_hour = int(session["open"].split(":")[0])
        close_hour = int(session["close"].split(":")[0])
        
        if open_hour < close_hour:
            is_open = open_hour <= current_hour < close_hour
        else:  # Session crosses midnight
            is_open = current_hour >= open_hour or current_hour < close_hour
        
        if is_open:
            current_sessions.append(session["name"])
    
    return {
        "current_time_utc": current_time,
        "active_sessions": current_sessions,
        "session_details": sessions,
        "overlaps": {
            "sydney_tokyo": "23:00-06:00 UTC",
            "tokyo_london": "07:00-08:00 UTC", 
            "london_new_york": "12:30-16:00 UTC"
        }
    }

@router.get("/economic-calendar")
async def get_economic_calendar(
    date: Optional[datetime] = None,
    importance: str = "high"
):
    """Get economic calendar events (mock data for demo)"""
    if not date:
        date = datetime.utcnow().date()
    
    # Mock economic events
    events = [
        {
            "time": "08:30",
            "currency": "USD",
            "event": "Non-Farm Payrolls",
            "importance": "high",
            "forecast": "180K",
            "previous": "175K",
            "impact": "bullish"
        },
        {
            "time": "10:00",
            "currency": "USD", 
            "event": "Unemployment Rate",
            "importance": "medium",
            "forecast": "3.7%",
            "previous": "3.8%",
            "impact": "bullish"
        },
        {
            "time": "14:00",
            "currency": "EUR",
            "event": "ECB Interest Rate Decision",
            "importance": "high",
            "forecast": "4.50%",
            "previous": "4.25%",
            "impact": "hawkish"
        }
    ]
    
    # Filter by importance
    if importance != "all":
        events = [e for e in events if e["importance"] == importance]
    
    return {
        "date": date,
        "events": events,
        "total_events": len(events),
        "high_impact_count": len([e for e in events if e["importance"] == "high"])
    }