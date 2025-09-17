import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
from dataclasses import dataclass, asdict

from app.models import (
    BacktestRequest, BacktestResult, TradeResult, 
    TradeSetup, TimeFrame, TradeDirection
)
from strategies.ict_strategies import ICTStrategyManager

@dataclass
class BacktestTrade:
    id: str
    setup_id: str
    symbol: str
    direction: TradeDirection
    entry_time: datetime
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: List[float]
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    status: str = "pending"
    commission: float = 0.0
    slippage: float = 0.0

class BacktestEngine:
    """Comprehensive backtesting engine for ICT strategies"""
    
    def __init__(self):
        self.strategy_manager = ICTStrategyManager()
        self.results_cache = {}
    
    async def run_backtest(self, request: BacktestRequest) -> BacktestResult:
        """Run a complete backtest with the specified parameters"""
        try:
            # Generate unique ID for this backtest
            backtest_id = str(uuid.uuid4())
            
            # Get historical data
            data = await self._get_historical_data(
                request.symbol, 
                request.timeframe,
                request.start_date,
                request.end_date
            )
            
            # Initialize portfolio
            portfolio = BacktestPortfolio(
                initial_capital=request.initial_capital,
                risk_per_trade=request.risk_per_trade
            )
            
            # Initialize trade tracking
            all_trades = []
            equity_curve = []
            
            # Run backtest day by day
            current_date = request.start_date
            while current_date < request.end_date:
                # Get data for current day
                daily_data = self._get_daily_data(data, current_date)
                
                if len(daily_data) == 0:
                    current_date += timedelta(days=1)
                    continue
                
                # Check for trade exits first
                portfolio = await self._check_trade_exits(portfolio, daily_data)
                
                # Generate new setups for current date
                for strategy_name in request.strategies:
                    # Get data up to current date for strategy analysis
                    historical_data = data[data['timestamp'] <= current_date + timedelta(days=1)]
                    
                    if len(historical_data) >= 20:  # Need minimum data for analysis
                        setups = await self.strategy_manager.get_trade_setups(
                            request.symbol, request.timeframe, [strategy_name]
                        )
                        
                        # Filter setups for current date (within the last day)
                        daily_setups = [s for s in setups 
                                      if abs((s.timestamp - current_date).days) <= 1]
                        
                        # Execute valid setups
                        for setup in daily_setups:
                            if portfolio.can_take_trade(setup, request.risk_per_trade):
                                trade = await self._execute_trade(setup, portfolio, daily_data.iloc[-1])
                                if trade:
                                    portfolio.add_trade(trade)
                                    all_trades.append(trade)
                
                # Record daily equity
                daily_equity = portfolio.get_current_equity()
                equity_curve.append({
                    "date": current_date,
                    "equity": daily_equity,
                    "drawdown": portfolio.get_current_drawdown(),
                    "open_trades": len(portfolio.open_trades)
                })
                
                current_date += timedelta(days=1)
            
            # Close any remaining open trades
            portfolio = await self._close_all_trades(portfolio, data.iloc[-1])
            
            # Calculate final metrics
            result = self._calculate_backtest_result(
                backtest_id=backtest_id,
                request=request,
                trades=all_trades,
                equity_curve=equity_curve,
                final_equity=portfolio.get_current_equity()
            )
            
            # Cache result
            self.results_cache[backtest_id] = result
            
            return result
            
        except Exception as e:
            raise Exception(f"Backtest execution failed: {str(e)}")
    
    async def get_backtest_result(self, backtest_id: str) -> Optional[BacktestResult]:
        """Retrieve cached backtest result"""
        return self.results_cache.get(backtest_id)
    
    async def list_backtest_results(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """List recent backtest results"""
        results = list(self.results_cache.values())
        
        # Sort by creation date (most recent first)
        results.sort(key=lambda x: x.created_at, reverse=True)
        
        # Apply pagination
        paginated_results = results[offset:offset + limit]
        
        # Return summary information
        return [
            {
                "id": result.id,
                "symbol": result.request.symbol,
                "timeframe": result.request.timeframe,
                "start_date": result.request.start_date,
                "end_date": result.request.end_date,
                "strategies": result.request.strategies,
                "total_trades": result.total_trades,
                "win_rate": result.win_rate,
                "total_pnl_percentage": result.total_pnl_percentage,
                "max_drawdown_percentage": result.max_drawdown_percentage,
                "created_at": result.created_at
            }
            for result in paginated_results
        ]
    
    async def optimize_strategy(
        self,
        symbol: str,
        strategy: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: TimeFrame,
        parameter_ranges: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Optimize strategy parameters using grid search"""
        
        optimization_results = []
        best_result = None
        best_score = float('-inf')
        
        # Generate parameter combinations
        param_combinations = self._generate_parameter_combinations(parameter_ranges)
        
        for i, params in enumerate(param_combinations):
            try:
                # Create backtest request with current parameters
                request = BacktestRequest(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe,
                    strategies=[strategy],
                    parameters=params
                )
                
                # Run backtest
                result = await self.run_backtest(request)
                
                # Calculate optimization score (e.g., Sharpe ratio adjusted for drawdown)
                score = self._calculate_optimization_score(result)
                
                optimization_result = {
                    "parameters": params,
                    "score": score,
                    "metrics": {
                        "total_trades": result.total_trades,
                        "win_rate": result.win_rate,
                        "total_pnl_percentage": result.total_pnl_percentage,
                        "max_drawdown_percentage": result.max_drawdown_percentage,
                        "sharpe_ratio": result.sharpe_ratio,
                        "profit_factor": result.profit_factor
                    }
                }
                
                optimization_results.append(optimization_result)
                
                # Track best result
                if score > best_score:
                    best_score = score
                    best_result = optimization_result
                
            except Exception as e:
                # Skip this parameter combination if it fails
                continue
        
        # Sort results by score
        optimization_results.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "strategy": strategy,
            "symbol": symbol,
            "optimization_period": f"{start_date.date()} to {end_date.date()}",
            "total_combinations_tested": len(optimization_results),
            "best_parameters": best_result["parameters"] if best_result else None,
            "best_score": best_score,
            "best_metrics": best_result["metrics"] if best_result else None,
            "all_results": optimization_results[:10],  # Top 10 results
            "parameter_ranges": parameter_ranges
        }
    
    # Private helper methods
    async def _get_historical_data(
        self, 
        symbol: str, 
        timeframe: TimeFrame, 
        start_date: datetime, 
        end_date: datetime
    ) -> pd.DataFrame:
        """Get historical market data for backtesting"""
        # Use strategy manager to get data
        days = (end_date - start_date).days
        return await self.strategy_manager._get_market_data(symbol, timeframe, days)
    
    def _get_daily_data(self, data: pd.DataFrame, date: datetime) -> pd.DataFrame:
        """Extract data for a specific date"""
        if 'timestamp' in data.columns:
            return data[data['timestamp'].dt.date == date.date()]
        return pd.DataFrame()
    
    async def _check_trade_exits(self, portfolio: 'BacktestPortfolio', data: pd.DataFrame) -> 'BacktestPortfolio':
        """Check and execute trade exits"""
        if len(data) == 0:
            return portfolio
        
        current_price = data['close'].iloc[-1]
        high_price = data['high'].iloc[-1]
        low_price = data['low'].iloc[-1]
        
        trades_to_remove = []
        
        for trade in portfolio.open_trades:
            exit_triggered = False
            exit_price = None
            exit_reason = None
            
            if trade.direction == TradeDirection.LONG:
                # Check stop loss
                if low_price <= trade.stop_loss:
                    exit_price = trade.stop_loss
                    exit_reason = "stop_loss"
                    exit_triggered = True
                # Check take profit
                elif high_price >= trade.take_profit[0]:
                    exit_price = trade.take_profit[0]
                    exit_reason = "take_profit"
                    exit_triggered = True
            
            else:  # SHORT
                # Check stop loss
                if high_price >= trade.stop_loss:
                    exit_price = trade.stop_loss
                    exit_reason = "stop_loss"
                    exit_triggered = True
                # Check take profit
                elif low_price <= trade.take_profit[0]:
                    exit_price = trade.take_profit[0]
                    exit_reason = "take_profit"
                    exit_triggered = True
            
            if exit_triggered:
                # Calculate PnL
                if trade.direction == TradeDirection.LONG:
                    pnl = (exit_price - trade.entry_price) * trade.quantity
                else:
                    pnl = (trade.entry_price - exit_price) * trade.quantity
                
                # Update trade
                trade.exit_time = data['timestamp'].iloc[-1]
                trade.exit_price = exit_price
                trade.exit_reason = exit_reason
                trade.pnl = pnl - trade.commission
                trade.pnl_percentage = (pnl / (trade.entry_price * trade.quantity)) * 100
                trade.status = "closed"
                
                # Update portfolio
                portfolio.close_trade(trade)
                trades_to_remove.append(trade)
        
        # Remove closed trades from open trades
        for trade in trades_to_remove:
            portfolio.open_trades.remove(trade)
        
        return portfolio
    
    async def _execute_trade(self, setup: TradeSetup, portfolio: 'BacktestPortfolio', current_bar: pd.Series = None) -> Optional[BacktestTrade]:
        """Execute a trade based on setup with current market conditions"""
        try:
            # Calculate position size based on risk
            risk_amount = portfolio.current_capital * portfolio.risk_per_trade
            price_distance = abs(setup.entry_price - setup.stop_loss)
            
            if price_distance == 0:
                return None
            
            quantity = risk_amount / price_distance
            
            # Adjust entry price based on current market conditions if available
            entry_price = setup.entry_price
            if current_bar is not None:
                current_price = current_bar['close']
                current_high = current_bar['high']
                current_low = current_bar['low']
                
                # Apply realistic slippage and execution logic
                if setup.direction == TradeDirection.LONG:
                    # For long trades, ensure we can actually buy at or near the setup price
                    if entry_price > current_high:
                        return None  # Can't execute above the high
                    if entry_price < current_low:
                        entry_price = current_low  # Fill at best available price
                    
                    # Add small slippage
                    slippage = entry_price * 0.0001  # 1 pip slippage
                    entry_price += slippage
                    
                else:  # SHORT
                    # For short trades, ensure we can actually sell at or near the setup price
                    if entry_price < current_low:
                        return None  # Can't execute below the low
                    if entry_price > current_high:
                        entry_price = current_high  # Fill at best available price
                    
                    # Add small slippage
                    slippage = entry_price * 0.0001  # 1 pip slippage
                    entry_price -= slippage
            
            # Recalculate position size with adjusted entry price
            adjusted_price_distance = abs(entry_price - setup.stop_loss)
            if adjusted_price_distance > 0:
                quantity = risk_amount / adjusted_price_distance
            else:
                return None
            
            # Calculate realistic commission
            commission = quantity * entry_price * 0.0002  # 0.02% commission (more realistic)
            
            # Create trade
            trade = BacktestTrade(
                id=str(uuid.uuid4()),
                setup_id=setup.id or str(uuid.uuid4()),
                symbol=setup.symbol,
                direction=setup.direction,
                entry_time=setup.timestamp,
                entry_price=entry_price,
                quantity=quantity,
                stop_loss=setup.stop_loss,
                take_profit=setup.take_profit,
                status="open",
                commission=commission,
                slippage=abs(entry_price - setup.entry_price) if current_bar is not None else 0
            )
            
            return trade
            
        except Exception as e:
            return None
    
    async def _close_all_trades(self, portfolio: 'BacktestPortfolio', final_bar: pd.Series) -> 'BacktestPortfolio':
        """Close all remaining open trades at the end of backtest"""
        final_price = final_bar['close']
        
        for trade in portfolio.open_trades:
            # Calculate PnL
            if trade.direction == TradeDirection.LONG:
                pnl = (final_price - trade.entry_price) * trade.quantity
            else:
                pnl = (trade.entry_price - final_price) * trade.quantity
            
            # Update trade
            trade.exit_time = final_bar['timestamp']
            trade.exit_price = final_price
            trade.exit_reason = "backtest_end"
            trade.pnl = pnl - trade.commission
            trade.pnl_percentage = (pnl / (trade.entry_price * trade.quantity)) * 100
            trade.status = "closed"
            
            portfolio.close_trade(trade)
        
        portfolio.open_trades.clear()
        return portfolio
    
    def _calculate_backtest_result(
        self,
        backtest_id: str,
        request: BacktestRequest,
        trades: List[BacktestTrade],
        equity_curve: List[Dict[str, Any]],
        final_equity: float
    ) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        
        # Convert trades to TradeResult format
        trade_results = []
        for trade in trades:
            result = TradeResult(
                id=trade.id,
                setup_id=trade.setup_id,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                pnl=trade.pnl,
                pnl_percentage=trade.pnl_percentage,
                status=trade.status
            )
            trade_results.append(result)
        
        # Calculate basic metrics
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.pnl and t.pnl > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate PnL metrics
        total_pnl = sum([t.pnl for t in trades if t.pnl])
        total_pnl_percentage = ((final_equity - request.initial_capital) / request.initial_capital) * 100
        
        # Calculate drawdown
        equity_values = [eq["equity"] for eq in equity_curve]
        max_drawdown = self._calculate_max_drawdown(equity_values)
        max_drawdown_percentage = (max_drawdown / request.initial_capital) * 100
        
        # Calculate other metrics
        wins = [t.pnl for t in trades if t.pnl and t.pnl > 0]
        losses = [abs(t.pnl) for t in trades if t.pnl and t.pnl < 0]
        
        average_win = np.mean(wins) if wins else 0
        average_loss = np.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = max(losses) if losses else 0
        
        # Calculate advanced metrics
        gross_profit = sum(wins) if wins else 0
        gross_loss = sum(losses) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calculate Sharpe ratio (simplified)
        returns = self._calculate_daily_returns(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        
        # Calculate monthly returns
        monthly_returns = self._calculate_monthly_returns(equity_curve)
        
        # Create result
        result = BacktestResult(
            id=backtest_id,
            request=request,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_pnl_percentage=total_pnl_percentage,
            max_drawdown=max_drawdown,
            max_drawdown_percentage=max_drawdown_percentage,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            trades=trade_results,
            equity_curve=equity_curve,
            monthly_returns=monthly_returns,
            created_at=datetime.utcnow()
        )
        
        return result
    
    def _calculate_max_drawdown(self, equity_values: List[float]) -> float:
        """Calculate maximum drawdown"""
        if not equity_values:
            return 0
        
        peak = equity_values[0]
        max_dd = 0
        
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = peak - value
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def _calculate_daily_returns(self, equity_curve: List[Dict[str, Any]]) -> List[float]:
        """Calculate daily returns from equity curve"""
        if len(equity_curve) < 2:
            return []
        
        returns = []
        for i in range(1, len(equity_curve)):
            prev_equity = equity_curve[i-1]["equity"]
            curr_equity = equity_curve[i]["equity"]
            daily_return = (curr_equity - prev_equity) / prev_equity
            returns.append(daily_return)
        
        return returns
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns:
            return 0
        
        avg_return = np.mean(returns) * 252  # Annualize
        volatility = np.std(returns) * np.sqrt(252)  # Annualize
        
        if volatility == 0:
            return 0
        
        return (avg_return - risk_free_rate) / volatility
    
    def _calculate_monthly_returns(self, equity_curve: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate monthly returns"""
        monthly_returns = {}
        
        if not equity_curve:
            return monthly_returns
        
        # Group by month
        monthly_data = {}
        for point in equity_curve:
            month_key = point["date"].strftime("%Y-%m")
            if month_key not in monthly_data:
                monthly_data[month_key] = []
            monthly_data[month_key].append(point["equity"])
        
        # Calculate returns for each month
        for month, equity_values in monthly_data.items():
            if len(equity_values) > 1:
                start_equity = equity_values[0]
                end_equity = equity_values[-1]
                monthly_return = ((end_equity - start_equity) / start_equity) * 100
                monthly_returns[month] = monthly_return
        
        return monthly_returns
    
    def _generate_parameter_combinations(self, parameter_ranges: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters for optimization"""
        # Simplified implementation - would need more sophisticated approach for real optimization
        combinations = []
        
        # For demo, generate a few sample combinations
        for i in range(5):
            combo = {}
            for param_name, param_range in parameter_ranges.items():
                if param_range.get("type") == "float":
                    min_val = param_range.get("min", 0)
                    max_val = param_range.get("max", 1)
                    combo[param_name] = min_val + (max_val - min_val) * (i / 4)
                elif param_range.get("type") == "int":
                    min_val = param_range.get("min", 1)
                    max_val = param_range.get("max", 10)
                    combo[param_name] = min_val + int((max_val - min_val) * (i / 4))
            combinations.append(combo)
        
        return combinations
    
    def _calculate_optimization_score(self, result: BacktestResult) -> float:
        """Calculate optimization score for parameter tuning"""
        # Combine multiple metrics into a single score
        # Higher is better
        
        if result.total_trades == 0:
            return float('-inf')
        
        # Base score from return
        return_score = result.total_pnl_percentage / 100
        
        # Penalty for high drawdown
        drawdown_penalty = result.max_drawdown_percentage / 100
        
        # Bonus for high win rate
        win_rate_bonus = (result.win_rate - 50) / 100  # Bonus above 50%
        
        # Bonus for good profit factor
        pf_bonus = min(result.profit_factor / 2, 1) if result.profit_factor != float('inf') else 1
        
        # Penalty for too few trades
        trade_count_factor = min(result.total_trades / 50, 1)  # Prefer at least 50 trades
        
        score = (return_score - drawdown_penalty + win_rate_bonus + pf_bonus) * trade_count_factor
        
        return score


class BacktestPortfolio:
    """Portfolio management for backtesting"""
    
    def __init__(self, initial_capital: float, risk_per_trade: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.risk_per_trade = risk_per_trade
        self.open_trades: List[BacktestTrade] = []
        self.closed_trades: List[BacktestTrade] = []
        self.peak_equity = initial_capital
    
    def can_take_trade(self, setup: TradeSetup, risk_per_trade: float) -> bool:
        """Check if portfolio can take another trade"""
        # Simple risk management - limit open trades and ensure sufficient capital
        if len(self.open_trades) >= 5:  # Max 5 concurrent trades
            return False
        
        risk_amount = self.current_capital * risk_per_trade
        return risk_amount > 0
    
    def add_trade(self, trade: BacktestTrade):
        """Add new trade to portfolio"""
        self.open_trades.append(trade)
        # Reserve capital for this trade
        position_value = trade.entry_price * trade.quantity
        self.current_capital -= trade.commission
    
    def close_trade(self, trade: BacktestTrade):
        """Close a trade and update portfolio"""
        self.closed_trades.append(trade)
        
        # Update capital with PnL
        if trade.pnl:
            self.current_capital += trade.pnl
        
        # Update peak equity
        current_equity = self.get_current_equity()
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
    
    def get_current_equity(self, current_prices: Dict[str, float] = None) -> float:
        """Get current portfolio equity including unrealized PnL"""
        equity = self.current_capital
        
        # Add unrealized PnL from open trades
        for trade in self.open_trades:
            if current_prices and trade.symbol in current_prices:
                current_price = current_prices[trade.symbol]
                
                # Calculate unrealized PnL
                if trade.direction == TradeDirection.LONG:
                    unrealized_pnl = (current_price - trade.entry_price) * trade.quantity
                else:
                    unrealized_pnl = (trade.entry_price - current_price) * trade.quantity
                
                equity += unrealized_pnl
        
        return equity
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown from peak"""
        current_equity = self.get_current_equity()
        return self.peak_equity - current_equity