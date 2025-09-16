from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Dict, Any
from datetime import datetime
from app.models import BacktestRequest, BacktestResult, TimeFrame
from backtesting.engine import BacktestEngine
from backtesting.metrics import PerformanceMetrics

router = APIRouter()
backtest_engine = BacktestEngine()
performance_metrics = PerformanceMetrics()

@router.post("/run", response_model=BacktestResult)
async def run_backtest(
    request: BacktestRequest,
    background_tasks: BackgroundTasks
):
    """Run a comprehensive backtest with ICT strategies"""
    try:
        # Validate request
        if request.start_date >= request.end_date:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        if request.risk_per_trade > 0.1:
            raise HTTPException(status_code=400, detail="Risk per trade cannot exceed 10%")
        
        # Run backtest
        result = await backtest_engine.run_backtest(request)
        
        # Calculate performance metrics
        result = performance_metrics.calculate_metrics(result)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")

@router.get("/results/{backtest_id}")
async def get_backtest_results(backtest_id: str):
    """Get backtest results by ID"""
    try:
        result = await backtest_engine.get_backtest_result(backtest_id)
        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get results: {str(e)}")

@router.get("/results")
async def list_backtest_results(
    limit: int = 20,
    offset: int = 0
):
    """List recent backtest results"""
    try:
        results = await backtest_engine.list_backtest_results(limit, offset)
        return {"results": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list results: {str(e)}")

@router.post("/compare")
async def compare_strategies(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: TimeFrame,
    strategies: List[str],
    initial_capital: float = 10000
):
    """Compare multiple ICT strategies performance"""
    try:
        comparison_results = []
        
        for strategy in strategies:
            request = BacktestRequest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                initial_capital=initial_capital,
                strategies=[strategy]
            )
            
            result = await backtest_engine.run_backtest(request)
            result = performance_metrics.calculate_metrics(result)
            
            comparison_results.append({
                "strategy": strategy,
                "metrics": {
                    "total_trades": result.total_trades,
                    "win_rate": result.win_rate,
                    "total_pnl_percentage": result.total_pnl_percentage,
                    "max_drawdown_percentage": result.max_drawdown_percentage,
                    "sharpe_ratio": result.sharpe_ratio,
                    "profit_factor": result.profit_factor
                }
            })
        
        # Sort by total PnL percentage
        comparison_results.sort(key=lambda x: x["metrics"]["total_pnl_percentage"], reverse=True)
        
        return {
            "comparison": comparison_results,
            "best_strategy": comparison_results[0]["strategy"] if comparison_results else None,
            "summary": {
                "period": f"{start_date.date()} to {end_date.date()}",
                "symbol": symbol,
                "timeframe": timeframe,
                "strategies_tested": len(strategies)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Strategy comparison failed: {str(e)}")

@router.get("/performance-metrics")
async def get_performance_metrics_info():
    """Get information about available performance metrics"""
    return {
        "metrics": [
            {
                "name": "Win Rate",
                "description": "Percentage of winning trades",
                "formula": "Winning Trades / Total Trades * 100"
            },
            {
                "name": "Profit Factor",
                "description": "Ratio of gross profit to gross loss",
                "formula": "Gross Profit / Gross Loss"
            },
            {
                "name": "Sharpe Ratio",
                "description": "Risk-adjusted return measure",
                "formula": "(Portfolio Return - Risk Free Rate) / Portfolio Standard Deviation"
            },
            {
                "name": "Maximum Drawdown",
                "description": "Largest peak-to-trough decline",
                "formula": "(Peak Value - Trough Value) / Peak Value * 100"
            },
            {
                "name": "Average Win/Loss",
                "description": "Average profit per winning/losing trade",
                "formula": "Total Profit(Loss) / Number of Winning(Losing) Trades"
            },
            {
                "name": "Risk-Reward Ratio",
                "description": "Average win divided by average loss",
                "formula": "Average Win / Average Loss"
            }
        ]
    }

@router.post("/optimize")
async def optimize_strategy(
    symbol: str,
    strategy: str,
    start_date: datetime,
    end_date: datetime,
    timeframe: TimeFrame,
    parameter_ranges: Dict[str, Dict[str, Any]]
):
    """Optimize strategy parameters using grid search"""
    try:
        optimization_result = await backtest_engine.optimize_strategy(
            symbol=symbol,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe,
            parameter_ranges=parameter_ranges
        )
        
        return optimization_result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Strategy optimization failed: {str(e)}")

@router.get("/equity-curve/{backtest_id}")
async def get_equity_curve(backtest_id: str):
    """Get equity curve data for visualization"""
    try:
        result = await backtest_engine.get_backtest_result(backtest_id)
        if not result:
            raise HTTPException(status_code=404, detail="Backtest result not found")
        
        return {
            "equity_curve": result.equity_curve,
            "drawdown_curve": performance_metrics.calculate_drawdown_curve(result.equity_curve),
            "underwater_curve": performance_metrics.calculate_underwater_curve(result.equity_curve)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get equity curve: {str(e)}")