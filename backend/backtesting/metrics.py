import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from app.models import BacktestResult
import matplotlib.pyplot as plt
import seaborn as sns

class PerformanceMetrics:
    """Calculate comprehensive performance metrics for backtesting results"""
    
    def calculate_metrics(self, result: BacktestResult) -> BacktestResult:
        """Calculate and enhance backtest result with additional metrics"""
        
        # Enhanced metrics already calculated in engine
        # This method can add additional sophisticated metrics
        
        return result
    
    def calculate_drawdown_curve(self, equity_curve: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate drawdown curve for visualization"""
        if not equity_curve:
            return []
        
        drawdown_curve = []
        peak_equity = equity_curve[0]["equity"]
        
        for point in equity_curve:
            current_equity = point["equity"]
            
            # Update peak
            if current_equity > peak_equity:
                peak_equity = current_equity
            
            # Calculate drawdown
            drawdown = peak_equity - current_equity
            drawdown_percentage = (drawdown / peak_equity) * 100 if peak_equity > 0 else 0
            
            drawdown_curve.append({
                "date": point["date"],
                "drawdown": drawdown,
                "drawdown_percentage": drawdown_percentage,
                "peak_equity": peak_equity
            })
        
        return drawdown_curve
    
    def calculate_underwater_curve(self, equity_curve: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate underwater (drawdown) curve for visualization"""
        drawdown_curve = self.calculate_drawdown_curve(equity_curve)
        
        underwater_curve = []
        for point in drawdown_curve:
            underwater_curve.append({
                "date": point["date"],
                "underwater_percentage": -point["drawdown_percentage"]  # Negative for underwater chart
            })
        
        return underwater_curve
    
    def calculate_risk_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        
        metrics = {
            "volatility": np.std(returns_array) * np.sqrt(252),  # Annualized
            "downside_deviation": self._calculate_downside_deviation(returns_array),
            "max_consecutive_losses": self._calculate_max_consecutive_losses(returns_array),
            "value_at_risk_95": np.percentile(returns_array, 5),
            "expected_shortfall_95": np.mean(returns_array[returns_array <= np.percentile(returns_array, 5)]),
            "skewness": self._calculate_skewness(returns_array),
            "kurtosis": self._calculate_kurtosis(returns_array),
            "calmar_ratio": self._calculate_calmar_ratio(returns_array),
            "sortino_ratio": self._calculate_sortino_ratio(returns_array)
        }
        
        return metrics
    
    def calculate_trade_analysis(self, trades: List[Any]) -> Dict[str, Any]:
        """Detailed trade analysis"""
        if not trades:
            return {}
        
        # Extract PnL values
        pnls = [trade.pnl for trade in trades if trade.pnl is not None]
        winning_trades = [pnl for pnl in pnls if pnl > 0]
        losing_trades = [pnl for pnl in pnls if pnl < 0]
        
        analysis = {
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": (len(winning_trades) / len(trades)) * 100 if trades else 0,
            "average_win": np.mean(winning_trades) if winning_trades else 0,
            "average_loss": np.mean(losing_trades) if losing_trades else 0,
            "largest_win": max(winning_trades) if winning_trades else 0,
            "largest_loss": min(losing_trades) if losing_trades else 0,
            "profit_factor": sum(winning_trades) / abs(sum(losing_trades)) if losing_trades else float('inf'),
            "expectancy": np.mean(pnls) if pnls else 0,
            "standard_deviation": np.std(pnls) if pnls else 0,
            "best_trade": max(pnls) if pnls else 0,
            "worst_trade": min(pnls) if pnls else 0
        }
        
        # Trade duration analysis
        durations = []
        for trade in trades:
            if trade.entry_time and trade.exit_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds() / 3600  # Hours
                durations.append(duration)
        
        if durations:
            analysis.update({
                "average_trade_duration_hours": np.mean(durations),
                "median_trade_duration_hours": np.median(durations),
                "shortest_trade_hours": min(durations),
                "longest_trade_hours": max(durations)
            })
        
        # Consecutive wins/losses
        analysis.update({
            "max_consecutive_wins": self._calculate_max_consecutive_wins(pnls),
            "max_consecutive_losses": self._calculate_max_consecutive_losses(pnls),
            "average_consecutive_wins": self._calculate_average_consecutive_wins(pnls),
            "average_consecutive_losses": self._calculate_average_consecutive_losses(pnls)
        })
        
        return analysis
    
    def calculate_monthly_statistics(self, monthly_returns: Dict[str, float]) -> Dict[str, Any]:
        """Calculate monthly performance statistics"""
        if not monthly_returns:
            return {}
        
        returns = list(monthly_returns.values())
        
        stats = {
            "total_months": len(returns),
            "positive_months": len([r for r in returns if r > 0]),
            "negative_months": len([r for r in returns if r < 0]),
            "best_month": max(returns) if returns else 0,
            "worst_month": min(returns) if returns else 0,
            "average_monthly_return": np.mean(returns) if returns else 0,
            "median_monthly_return": np.median(returns) if returns else 0,
            "monthly_volatility": np.std(returns) if returns else 0,
            "monthly_sharpe": self._calculate_monthly_sharpe(returns),
            "months_above_benchmark": len([r for r in returns if r > 2.0])  # Above 2% monthly
        }
        
        return stats
    
    def generate_performance_report(self, result: BacktestResult) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        # Calculate daily returns
        daily_returns = self._extract_daily_returns(result.equity_curve)
        
        # Risk metrics
        risk_metrics = self.calculate_risk_metrics(daily_returns)
        
        # Trade analysis
        trade_analysis = self.calculate_trade_analysis(result.trades)
        
        # Monthly statistics
        monthly_stats = self.calculate_monthly_statistics(result.monthly_returns)
        
        # Benchmark comparison (simplified - comparing to market return)
        benchmark_comparison = self._compare_to_benchmark(result, 0.08)  # 8% annual return
        
        report = {
            "summary": {
                "strategy": result.request.strategies,
                "symbol": result.request.symbol,
                "period": f"{result.request.start_date.date()} to {result.request.end_date.date()}",
                "initial_capital": result.request.initial_capital,
                "final_equity": result.request.initial_capital + result.total_pnl,
                "total_return": result.total_pnl_percentage,
                "annualized_return": self._annualize_return(result.total_pnl_percentage, result.request.start_date, result.request.end_date),
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "max_drawdown": result.max_drawdown_percentage,
                "sharpe_ratio": result.sharpe_ratio
            },
            "risk_metrics": risk_metrics,
            "trade_analysis": trade_analysis,
            "monthly_statistics": monthly_stats,
            "benchmark_comparison": benchmark_comparison,
            "key_insights": self._generate_key_insights(result, risk_metrics, trade_analysis)
        }
        
        return report
    
    # Private helper methods
    def _calculate_downside_deviation(self, returns: np.ndarray, target: float = 0) -> float:
        """Calculate downside deviation"""
        downside_returns = returns[returns < target]
        return np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
    
    def _calculate_max_consecutive_losses(self, returns: np.ndarray) -> int:
        """Calculate maximum consecutive losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for ret in returns:
            if ret < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        if len(returns) < 3:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
        return skewness
    
    def _calculate_kurtosis(self, returns: np.ndarray) -> float:
        """Calculate kurtosis of returns"""
        if len(returns) < 4:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        kurtosis = np.mean(((returns - mean_return) / std_return) ** 4) - 3
        return kurtosis
    
    def _calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annual_return = np.mean(returns) * 252
        max_dd = self._calculate_max_drawdown_from_returns(returns)
        
        return annual_return / max_dd if max_dd != 0 else 0
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, target: float = 0) -> float:
        """Calculate Sortino ratio"""
        excess_return = np.mean(returns) - target/252
        downside_dev = self._calculate_downside_deviation(returns, target/252)
        
        return (excess_return * 252) / downside_dev if downside_dev != 0 else 0
    
    def _calculate_max_drawdown_from_returns(self, returns: np.ndarray) -> float:
        """Calculate max drawdown from returns series"""
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        return np.max(drawdown)
    
    def _calculate_max_consecutive_wins(self, pnls: List[float]) -> int:
        """Calculate maximum consecutive wins"""
        max_consecutive = 0
        current_consecutive = 0
        
        for pnl in pnls:
            if pnl > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_average_consecutive_wins(self, pnls: List[float]) -> float:
        """Calculate average consecutive wins"""
        consecutive_sequences = []
        current_consecutive = 0
        
        for pnl in pnls:
            if pnl > 0:
                current_consecutive += 1
            else:
                if current_consecutive > 0:
                    consecutive_sequences.append(current_consecutive)
                current_consecutive = 0
        
        if current_consecutive > 0:
            consecutive_sequences.append(current_consecutive)
        
        return np.mean(consecutive_sequences) if consecutive_sequences else 0
    
    def _calculate_average_consecutive_losses(self, pnls: List[float]) -> float:
        """Calculate average consecutive losses"""
        consecutive_sequences = []
        current_consecutive = 0
        
        for pnl in pnls:
            if pnl < 0:
                current_consecutive += 1
            else:
                if current_consecutive > 0:
                    consecutive_sequences.append(current_consecutive)
                current_consecutive = 0
        
        if current_consecutive > 0:
            consecutive_sequences.append(current_consecutive)
        
        return np.mean(consecutive_sequences) if consecutive_sequences else 0
    
    def _calculate_monthly_sharpe(self, returns: List[float]) -> float:
        """Calculate monthly Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        # Assume 2% annual risk-free rate (0.167% monthly)
        risk_free_monthly = 0.02 / 12
        return (mean_return - risk_free_monthly) / std_return
    
    def _extract_daily_returns(self, equity_curve: List[Dict[str, Any]]) -> List[float]:
        """Extract daily returns from equity curve"""
        if len(equity_curve) < 2:
            return []
        
        returns = []
        for i in range(1, len(equity_curve)):
            prev_equity = equity_curve[i-1]["equity"]
            curr_equity = equity_curve[i]["equity"]
            
            if prev_equity != 0:
                daily_return = (curr_equity - prev_equity) / prev_equity
                returns.append(daily_return)
        
        return returns
    
    def _compare_to_benchmark(self, result: BacktestResult, benchmark_annual_return: float) -> Dict[str, Any]:
        """Compare strategy performance to benchmark"""
        strategy_annual_return = self._annualize_return(
            result.total_pnl_percentage, 
            result.request.start_date, 
            result.request.end_date
        )
        
        excess_return = strategy_annual_return - benchmark_annual_return
        
        return {
            "benchmark_annual_return": benchmark_annual_return,
            "strategy_annual_return": strategy_annual_return,
            "excess_return": excess_return,
            "outperformed": excess_return > 0,
            "information_ratio": excess_return / (result.sharpe_ratio * np.sqrt(252)) if result.sharpe_ratio != 0 else 0
        }
    
    def _annualize_return(self, total_return: float, start_date, end_date) -> float:
        """Annualize total return"""
        days = (end_date - start_date).days
        years = days / 365.25
        
        if years == 0:
            return 0
        
        return ((1 + total_return/100) ** (1/years) - 1) * 100
    
    def _generate_key_insights(self, result: BacktestResult, risk_metrics: Dict[str, Any], trade_analysis: Dict[str, Any]) -> List[str]:
        """Generate key insights from backtest results"""
        insights = []
        
        # Performance insights
        if result.total_pnl_percentage > 20:
            insights.append(f"Strong performance with {result.total_pnl_percentage:.1f}% total return")
        elif result.total_pnl_percentage > 10:
            insights.append(f"Moderate performance with {result.total_pnl_percentage:.1f}% total return")
        else:
            insights.append(f"Underperformed with {result.total_pnl_percentage:.1f}% total return")
        
        # Win rate insights
        if result.win_rate > 60:
            insights.append(f"High win rate of {result.win_rate:.1f}% indicates good entry timing")
        elif result.win_rate < 40:
            insights.append(f"Low win rate of {result.win_rate:.1f}% suggests entry timing needs improvement")
        
        # Risk insights
        if result.max_drawdown_percentage > 20:
            insights.append(f"High maximum drawdown of {result.max_drawdown_percentage:.1f}% indicates high risk")
        elif result.max_drawdown_percentage < 10:
            insights.append(f"Low maximum drawdown of {result.max_drawdown_percentage:.1f}% shows good risk control")
        
        # Profit factor insights
        if result.profit_factor > 2:
            insights.append(f"Excellent profit factor of {result.profit_factor:.2f}")
        elif result.profit_factor > 1.5:
            insights.append(f"Good profit factor of {result.profit_factor:.2f}")
        elif result.profit_factor < 1:
            insights.append("Negative profit factor indicates more losses than profits")
        
        # Trade count insights
        if result.total_trades < 20:
            insights.append("Low trade count may indicate limited opportunities or overly strict criteria")
        elif result.total_trades > 100:
            insights.append("High trade count provides good statistical significance")
        
        return insights