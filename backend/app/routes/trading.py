from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import (
    TradeSetup, TradeResult, MarketAnalysis, 
    TimeFrame, TradeDirection, ICTConcept
)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from strategies.ict_strategies import ICTStrategyManager

router = APIRouter()
strategy_manager = ICTStrategyManager()

@router.get("/strategies")
async def get_available_strategies():
    """Get list of available ICT trading strategies"""
    return {
        "strategies": [
            {
                "name": "order_block_strategy",
                "description": "Trade order blocks with proper mitigation",
                "concepts": ["order_blocks", "market_structure"]
            },
            {
                "name": "fair_value_gap_strategy", 
                "description": "Trade fair value gaps and imbalances",
                "concepts": ["fair_value_gaps", "inefficiencies"]
            },
            {
                "name": "silver_bullet_strategy",
                "description": "ICT Silver Bullet 15-min NY Open strategy",
                "concepts": ["killzones", "liquidity_raids"]
            },
            {
                "name": "breaker_block_strategy",
                "description": "Trade breaker blocks after structure break",
                "concepts": ["breaker_blocks", "market_structure"]
            },
            {
                "name": "liquidity_raid_strategy",
                "description": "Trade liquidity raids and reversals",
                "concepts": ["liquidity_pools", "stop_hunts"]
            },
            {
                "name": "smt_divergence_strategy",
                "description": "Smart Money Divergence across pairs",
                "concepts": ["smt_divergence", "correlation"]
            },
            {
                "name": "power_of_three_strategy",
                "description": "Accumulation-Manipulation-Distribution model",
                "concepts": ["power_of_3", "market_phases"]
            }
        ]
    }

@router.get("/concepts")
async def get_ict_concepts():
    """Get comprehensive list of ICT concepts implemented"""
    return {
        "core_concepts": [
            "Market Structure (HH, HL, LH, LL)",
            "Liquidity (buy-side & sell-side)",
            "Liquidity Pools (equal highs/lows, trendline liquidity)",
            "Order Blocks (Bullish & Bearish)",
            "Breaker Blocks",
            "Fair Value Gaps (FVG) / Imbalances",
            "Rejection Blocks",
            "Mitigation Blocks",
            "Supply & Demand Zones",
            "Premium & Discount (Optimal Trade Entry - OTE)",
            "Dealing Ranges",
            "Swing Highs & Swing Lows",
            "Market Maker Buy & Sell Models",
            "Judas Swing (false breakout at sessions open)",
            "Turtle Soup (stop-hunt strategy)",
            "Power of 3 (Accumulation – Manipulation – Distribution)",
            "Optimal Trade Entry (retracement into 62%-79% zone)",
            "SMT Divergence (Smart Money Divergence)",
            "Liquidity Voids / Inefficiencies"
        ],
        "time_price_theory": [
            "Killzones (London Open, New York Open, London Close, Asia Range)",
            "Midnight Open / Session Opens",
            "Equilibrium & Fibonacci Ratios (50%, 62%, 70.5%, 79%)",
            "Daily & Weekly Range Expectations",
            "Session Liquidity Raids",
            "Weekly Profiles (WHLC)",
            "Daily Bias (using daily open, previous day's high/low)",
            "Weekly Bias (using weekly OHLC)",
            "Monthly Bias (using monthly OHLC)",
            "Time of Day Highs & Lows (AM/PM session separation)"
        ],
        "advanced_concepts": [
            "High Probability Trade Scenarios",
            "Liquidity Runs (stop hunts, inducement, fakeouts)",
            "Reversals vs. Continuations",
            "Accumulation & Distribution Schematics",
            "Order Flow (institutional narrative)",
            "High/Low of the Day Identification",
            "Range Expansion (daily/weekly breakouts)",
            "Inside Day / Outside Day concepts",
            "Weekly Profiles (expansion, consolidation, reversal)",
            "Interbank Price Delivery Algorithm (IPDA) theory",
            "Algo-based Price Delivery"
        ]
    }

@router.post("/analyze")
async def analyze_market(
    symbol: str,
    timeframe: TimeFrame = TimeFrame.H1,
    lookback_days: int = 30
):
    """Analyze market using ICT concepts"""
    try:
        analysis = await strategy_manager.analyze_market(symbol, timeframe, lookback_days)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/setups")
async def get_trade_setups(
    symbol: str,
    timeframe: TimeFrame = TimeFrame.H1,
    strategies: List[str] = None
):
    """Get current trade setups based on ICT strategies"""
    try:
        if strategies is None:
            strategies = ["order_block_strategy", "fair_value_gap_strategy"]
        
        setups = await strategy_manager.get_trade_setups(symbol, timeframe, strategies)
        return {"setups": setups}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Setup generation failed: {str(e)}")

@router.get("/market-structure/{symbol}")
async def get_market_structure(
    symbol: str,
    timeframe: TimeFrame = TimeFrame.H4
):
    """Get current market structure analysis"""
    try:
        structure = await strategy_manager.get_market_structure(symbol, timeframe)
        return structure
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market structure analysis failed: {str(e)}")

@router.get("/killzones")
async def get_killzones():
    """Get ICT killzone information"""
    return {
        "killzones": [
            {
                "name": "London Open Killzone",
                "time_utc": "07:00-10:00",
                "description": "European session open, high volatility period"
            },
            {
                "name": "New York AM Killzone", 
                "time_utc": "12:30-15:00",
                "description": "US session open, major moves expected"
            },
            {
                "name": "New York PM Killzone",
                "time_utc": "15:00-18:00", 
                "description": "Afternoon session, potential reversals"
            },
            {
                "name": "London Close Killzone",
                "time_utc": "15:00-16:00",
                "description": "European session close, profit taking"
            },
            {
                "name": "Asia Range",
                "time_utc": "20:00-06:00",
                "description": "Low volatility consolidation period"
            }
        ],
        "current_killzone": strategy_manager.get_current_killzone()
    }