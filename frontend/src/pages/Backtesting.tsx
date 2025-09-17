import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Button,
  Card,
  CardContent,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
} from '@mui/material';
import { Line, Bar } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend
);

interface BacktestResult {
  id: string;
  strategy: string;
  symbol: string;
  timeframe: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
  finalCapital: number;
  totalReturn: number;
  winRate: number;
  maxDrawdown: number;
  sharpeRatio: number;
  totalTrades: number;
  status: 'completed' | 'running' | 'failed';
}

interface Trade {
  id: string;
  timestamp: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  entry: number;
  exit: number;
  quantity: number;
  pnl: number;
  strategy: string;
}

const Backtesting: React.FC = () => {
  const [selectedStrategy, setSelectedStrategy] = useState('order_block');
  const [selectedSymbol, setSelectedSymbol] = useState('EURUSD');
  const [customSymbol, setCustomSymbol] = useState('');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [isRunning, setIsRunning] = useState(false);
  
  // All available strategies from backend
  const availableStrategies = [
    'market_structure_strategy',
    'liquidity_strategy', 
    'order_block_strategy',
    'breaker_block_strategy',
    'fair_value_gap_strategy',
    'rejection_block_strategy',
    'mitigation_block_strategy',
    'supply_demand_zones_strategy',
    'premium_discount_strategy',
    'liquidity_pools_strategy',
    'dealing_ranges_strategy',
    'swing_points_strategy',
    'market_maker_models_strategy',
    'judas_swing_strategy',
    'turtle_soup_strategy',
    'power_of_three_strategy',
    'optimal_trade_entry_strategy',
    'smt_divergence_strategy',
    'liquidity_voids_strategy',
    'killzones_strategy',
    'session_opens_strategy',
    'fibonacci_ratios_strategy',
    'range_expectations_strategy',
    'session_liquidity_raids_strategy',
    'weekly_profiles_strategy',
    'daily_bias_strategy',
    'weekly_bias_strategy',
    'monthly_bias_strategy',
    'time_of_day_strategy'
  ];
  
  // Available symbols
  const availableSymbols = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
    'IRFC.NS', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFC.NS'
  ];
  
  const handleCustomSymbolAdd = () => {
    if (customSymbol.trim() && !availableSymbols.includes(customSymbol.toUpperCase())) {
      setSelectedSymbol(customSymbol.toUpperCase());
      setCustomSymbol('');
    }
  };
  
  const [backtestResults, setBacktestResults] = useState<BacktestResult[]>([
    {
      id: '1',
      strategy: 'Order Block Strategy',
      symbol: 'EURUSD',
      timeframe: '1h',
      startDate: '2023-01-01',
      endDate: '2023-12-31',
      initialCapital: 10000,
      finalCapital: 14567,
      totalReturn: 45.67,
      winRate: 68.5,
      maxDrawdown: -12.3,
      sharpeRatio: 1.85,
      totalTrades: 147,
      status: 'completed',
    },
    {
      id: '2',
      strategy: 'Fair Value Gap',
      symbol: 'GBPUSD',
      timeframe: '4h',
      startDate: '2023-01-01',
      endDate: '2023-12-31',
      initialCapital: 10000,
      finalCapital: 12890,
      totalReturn: 28.9,
      winRate: 72.1,
      maxDrawdown: -8.7,
      sharpeRatio: 1.92,
      totalTrades: 98,
      status: 'completed',
    },
    {
      id: '3',
      strategy: 'Market Structure',
      symbol: 'USDJPY',
      timeframe: '1h',
      startDate: '2023-01-01',
      endDate: '2023-12-31',
      initialCapital: 10000,
      finalCapital: 13245,
      totalReturn: 32.45,
      winRate: 65.8,
      maxDrawdown: -15.2,
      sharpeRatio: 1.67,
      totalTrades: 203,
      status: 'completed',
    },
  ]);

  const [trades, setTrades] = useState<Trade[]>([
    {
      id: '1',
      timestamp: '2023-12-15 14:30:00',
      symbol: 'EURUSD',
      side: 'BUY',
      entry: 1.0835,
      exit: 1.0867,
      quantity: 10000,
      pnl: 320,
      strategy: 'Order Block',
    },
    {
      id: '2',
      timestamp: '2023-12-15 16:45:00',
      symbol: 'EURUSD',
      side: 'SELL',
      entry: 1.0851,
      exit: 1.0829,
      quantity: 8000,
      pnl: 176,
      strategy: 'Order Block',
    },
    {
      id: '3',
      timestamp: '2023-12-15 18:20:00',
      symbol: 'EURUSD',
      side: 'BUY',
      entry: 1.0823,
      exit: 1.0809,
      quantity: 12000,
      pnl: -168,
      strategy: 'Order Block',
    },
  ]);

  const equityCurveData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    datasets: [
      {
        label: 'Portfolio Value',
        data: [10000, 10245, 10890, 11230, 11678, 12100, 12456, 12789, 13123, 13456, 13890, 14567],
        borderColor: '#00ff88',
        backgroundColor: 'rgba(0, 255, 136, 0.1)',
        borderWidth: 2,
        fill: true,
      },
      {
        label: 'Drawdown',
        data: [0, -2.3, -1.8, -3.2, -2.1, -4.5, -5.2, -3.8, -2.9, -6.7, -8.9, -4.2],
        borderColor: '#ff5722',
        backgroundColor: 'rgba(255, 87, 34, 0.1)',
        borderWidth: 2,
        fill: true,
      },
    ],
  };

  const monthlyReturnsData = {
    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    datasets: [
      {
        label: 'Monthly Returns (%)',
        data: [2.45, 6.26, 3.82, 4.08, 3.81, 2.32, 2.98, 2.70, 2.72, 2.67, 3.26, 4.96],
        backgroundColor: [
          '#00ff88', '#00ff88', '#00ff88', '#00ff88', '#00ff88', '#00ff88',
          '#00ff88', '#00ff88', '#00ff88', '#00ff88', '#00ff88', '#00ff88'
        ],
        borderColor: '#00ff88',
        borderWidth: 1,
      },
    ],
  };

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: '#ffffff',
        },
      },
    },
    scales: {
      x: {
        ticks: { color: '#b0b0b0' },
        grid: { color: '#333' },
      },
      y: {
        ticks: { color: '#b0b0b0' },
        grid: { color: '#333' },
      },
    },
  };

  const barChartOptions: ChartOptions<'bar'> = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: '#ffffff',
        },
      },
    },
    scales: {
      x: {
        ticks: { color: '#b0b0b0' },
        grid: { color: '#333' },
      },
      y: {
        ticks: { color: '#b0b0b0' },
        grid: { color: '#333' },
      },
    },
  };

  const runBacktest = async () => {
    setIsRunning(true);
    // Simulate backtest running
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Add new result
    const newResult: BacktestResult = {
      id: Date.now().toString(),
      strategy: strategies.find(s => s.value === selectedStrategy)?.label || '',
      symbol: selectedSymbol,
      timeframe: selectedTimeframe,
      startDate: '2023-01-01',
      endDate: '2023-12-31',
      initialCapital: 10000,
      finalCapital: 10000 + Math.random() * 5000,
      totalReturn: Math.random() * 50,
      winRate: 60 + Math.random() * 20,
      maxDrawdown: -(Math.random() * 20),
      sharpeRatio: 1 + Math.random(),
      totalTrades: Math.floor(50 + Math.random() * 200),
      status: 'completed',
    };
    
    setBacktestResults(prev => [newResult, ...prev]);
    setIsRunning(false);
  };

  const strategies = [
    // Core ICT Strategies
    { value: 'market_structure_strategy', label: 'Market Structure Strategy' },
    { value: 'liquidity_strategy', label: 'Liquidity Strategy' },
    { value: 'order_block_strategy', label: 'Order Block Strategy' },
    { value: 'breaker_block_strategy', label: 'Breaker Block Strategy' },
    { value: 'fair_value_gap_strategy', label: 'Fair Value Gap Strategy' },
    { value: 'rejection_block_strategy', label: 'Rejection Block Strategy' },
    { value: 'mitigation_block_strategy', label: 'Mitigation Block Strategy' },
    { value: 'supply_demand_zones_strategy', label: 'Supply & Demand Zones' },
    { value: 'premium_discount_strategy', label: 'Premium Discount Strategy' },
    { value: 'liquidity_pools_strategy', label: 'Liquidity Pools Strategy' },
    { value: 'dealing_ranges_strategy', label: 'Dealing Ranges Strategy' },
    { value: 'swing_points_strategy', label: 'Swing Points Strategy' },
    { value: 'market_maker_models_strategy', label: 'Market Maker Models' },
    { value: 'judas_swing_strategy', label: 'Judas Swing Strategy' },
    { value: 'turtle_soup_strategy', label: 'Turtle Soup Strategy' },
    { value: 'power_of_three_strategy', label: 'Power of Three Strategy' },
    { value: 'optimal_trade_entry_strategy', label: 'Optimal Trade Entry' },
    { value: 'smt_divergence_strategy', label: 'SMT Divergence Strategy' },
    { value: 'liquidity_voids_strategy', label: 'Liquidity Voids Strategy' },
    
    // Time & Price Theory
    { value: 'killzones_strategy', label: 'Killzones Strategy' },
    { value: 'session_opens_strategy', label: 'Session Opens Strategy' },
    { value: 'fibonacci_ratios_strategy', label: 'Fibonacci Ratios Strategy' },
    { value: 'range_expectations_strategy', label: 'Range Expectations Strategy' },
    { value: 'session_liquidity_raids_strategy', label: 'Session Liquidity Raids' },
    { value: 'weekly_profiles_strategy', label: 'Weekly Profiles Strategy' },
    { value: 'daily_bias_strategy', label: 'Daily Bias Strategy' },
    { value: 'weekly_bias_strategy', label: 'Weekly Bias Strategy' },
    { value: 'monthly_bias_strategy', label: 'Monthly Bias Strategy' },
    { value: 'time_of_day_strategy', label: 'Time of Day Strategy' },
  ];

  const symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD',
                   'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA',
                   'IRFC.NS', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFC.NS'];
  const timeframes = ['15m', '30m', '1h', '4h', '1d'];

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 3 }}>
        Backtesting Engine
      </Typography>

      {/* Backtest Configuration */}
      <Paper sx={{ p: 3, mb: 3, backgroundColor: 'background.paper' }}>
        <Typography variant="h6" gutterBottom>
          Configure Backtest
        </Typography>
        <Grid container spacing={3} alignItems="center">
          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
              <InputLabel>Strategy</InputLabel>
              <Select
                value={selectedStrategy}
                label="Strategy"
                onChange={(e) => setSelectedStrategy(e.target.value)}
              >
                {strategies.map((strategy) => (
                  <MenuItem key={strategy.value} value={strategy.value}>
                    {strategy.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
              <InputLabel>Symbol</InputLabel>
              <Select
                value={selectedSymbol}
                label="Symbol"
                onChange={(e) => setSelectedSymbol(e.target.value)}
              >
                {symbols.map((symbol) => (
                  <MenuItem key={symbol} value={symbol}>
                    {symbol}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <Box sx={{ display: 'flex', gap: 1 }}>
              <TextField
                size="small"
                placeholder="Custom Symbol"
                value={customSymbol}
                onChange={(e) => setCustomSymbol(e.target.value.toUpperCase())}
                sx={{ flex: 1 }}
              />
              <Button
                variant="outlined"
                size="small"
                onClick={handleCustomSymbolAdd}
                disabled={!customSymbol.trim()}
              >
                Add
              </Button>
            </Box>
          </Grid>
          <Grid item xs={12} md={2}>
            <FormControl fullWidth>
              <InputLabel>Timeframe</InputLabel>
              <Select
                value={selectedTimeframe}
                label="Timeframe"
                onChange={(e) => setSelectedTimeframe(e.target.value)}
              >
                {timeframes.map((tf) => (
                  <MenuItem key={tf} value={tf}>
                    {tf}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
          </Grid>
          <Grid item xs={12} md={2}>
            <Button
              variant="contained"
              onClick={runBacktest}
              disabled={isRunning}
              fullWidth
              sx={{ height: 56 }}
            >
              {isRunning ? 'Running...' : 'Run Backtest'}
            </Button>
          </Grid>
        </Grid>
      </Paper>

      {/* Performance Charts */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={8}>
          <Paper className="trading-chart" sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Equity Curve & Drawdown
            </Typography>
            <Line data={equityCurveData} options={chartOptions} />
          </Paper>
        </Grid>
        <Grid item xs={12} md={4}>
          <Paper className="trading-chart" sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Monthly Returns
            </Typography>
            <Bar data={monthlyReturnsData} options={barChartOptions} />
          </Paper>
        </Grid>
      </Grid>

      {/* Backtest Results Table */}
      <Paper sx={{ p: 3, mb: 3, backgroundColor: 'background.paper' }}>
        <Typography variant="h6" gutterBottom>
          Backtest Results
        </Typography>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Strategy</TableCell>
                <TableCell>Symbol</TableCell>
                <TableCell>Timeframe</TableCell>
                <TableCell align="right">Return %</TableCell>
                <TableCell align="right">Win Rate %</TableCell>
                <TableCell align="right">Max DD %</TableCell>
                <TableCell align="right">Sharpe</TableCell>
                <TableCell align="right">Trades</TableCell>
                <TableCell>Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {backtestResults.map((result) => (
                <TableRow key={result.id}>
                  <TableCell>{result.strategy}</TableCell>
                  <TableCell>{result.symbol}</TableCell>
                  <TableCell>{result.timeframe}</TableCell>
                  <TableCell align="right" className={result.totalReturn >= 0 ? 'profit' : 'loss'}>
                    {result.totalReturn.toFixed(2)}%
                  </TableCell>
                  <TableCell align="right">{result.winRate.toFixed(1)}%</TableCell>
                  <TableCell align="right" className="loss">
                    {result.maxDrawdown.toFixed(1)}%
                  </TableCell>
                  <TableCell align="right">{result.sharpeRatio.toFixed(2)}</TableCell>
                  <TableCell align="right">{result.totalTrades}</TableCell>
                  <TableCell>
                    <Chip
                      label={result.status}
                      color={result.status === 'completed' ? 'primary' : 'default'}
                      size="small"
                    />
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>

      {/* Trade Log */}
      <Paper sx={{ p: 3, backgroundColor: 'background.paper' }}>
        <Typography variant="h6" gutterBottom>
          Recent Trades (Order Block Strategy - EURUSD)
        </Typography>
        <TableContainer>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Timestamp</TableCell>
                <TableCell>Side</TableCell>
                <TableCell align="right">Entry</TableCell>
                <TableCell align="right">Exit</TableCell>
                <TableCell align="right">Quantity</TableCell>
                <TableCell align="right">P&L</TableCell>
                <TableCell>Strategy</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {trades.map((trade) => (
                <TableRow key={trade.id}>
                  <TableCell>{trade.timestamp}</TableCell>
                  <TableCell>
                    <Chip
                      label={trade.side}
                      color={trade.side === 'BUY' ? 'primary' : 'secondary'}
                      size="small"
                    />
                  </TableCell>
                  <TableCell align="right">{trade.entry.toFixed(4)}</TableCell>
                  <TableCell align="right">{trade.exit.toFixed(4)}</TableCell>
                  <TableCell align="right">{trade.quantity.toLocaleString()}</TableCell>
                  <TableCell align="right" className={trade.pnl >= 0 ? 'profit' : 'loss'}>
                    ${trade.pnl.toFixed(2)}
                  </TableCell>
                  <TableCell>{trade.strategy}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      </Paper>
    </Box>
  );
};

export default Backtesting;