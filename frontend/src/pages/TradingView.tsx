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
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  ShowChart,
  Timeline,
  Psychology,
  Speed,
} from '@mui/icons-material';
import { Line, Scatter } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
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
  Title,
  Tooltip,
  Legend
);

interface MarketData {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface OrderBlock {
  id: string;
  price: number;
  type: 'bullish' | 'bearish';
  strength: number;
  timestamp: string;
}

interface FVG {
  id: string;
  high: number;
  low: number;
  type: 'bullish' | 'bearish';
  timestamp: string;
}

const TradingView: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('EURUSD');
  const [selectedTimeframe, setSelectedTimeframe] = useState('1h');
  const [marketData, setMarketData] = useState<MarketData[]>([]);
  const [orderBlocks, setOrderBlocks] = useState<OrderBlock[]>([]);
  const [fvgs, setFvgs] = useState<FVG[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [priceChange, setPriceChange] = useState<number>(0);

  // Simulated market data
  useEffect(() => {
    generateMarketData();
    generateOrderBlocks();
    generateFVGs();
  }, [selectedSymbol, selectedTimeframe]);

  const generateMarketData = () => {
    const data: MarketData[] = [];
    let basePrice = getBasePriceForSymbol(selectedSymbol);
    const now = new Date();
    
    for (let i = 100; i >= 0; i--) {
      const timestamp = new Date(now.getTime() - i * 60 * 60 * 1000).toISOString();
      const change = (Math.random() - 0.5) * (basePrice * 0.02); // 2% max change
      const open = basePrice;
      const close = basePrice + change;
      const high = Math.max(open, close) + Math.random() * (basePrice * 0.005);
      const low = Math.min(open, close) - Math.random() * (basePrice * 0.005);
      
      data.push({
        timestamp,
        open,
        high,
        low,
        close,
        volume: Math.floor(Math.random() * 1000000),
      });
      
      basePrice = close;
    }
    
    setMarketData(data);
    if (data.length > 0) {
      const latest = data[data.length - 1];
      const previous = data[data.length - 2];
      setCurrentPrice(latest.close);
      setPriceChange(previous ? ((latest.close - previous.close) / previous.close) * 100 : 0);
    }
  };

  const getBasePriceForSymbol = (symbol: string): number => {
    const basePrices: { [key: string]: number } = {
      'EURUSD': 1.0800,
      'GBPUSD': 1.2650,
      'USDJPY': 148.50,
      'AUDUSD': 0.6850,
      'USDCAD': 1.3520,
      'IRFC.NS': 45.50,   // Indian Railway Finance Corporation
      'RELIANCE.NS': 2850.00,
      'TCS.NS': 3200.00,
      'INFY.NS': 1450.00,
      'HDFC.NS': 1680.00,
      'AAPL': 175.00,
      'TSLA': 240.00,
      'MSFT': 380.00,
      'GOOGL': 140.00,
    };
    return basePrices[symbol] || 100.00;
  };

  const runAIAnalysis = async () => {
    setIsLoading(true);
    // Simulate AI analysis
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Generate new analysis for current symbol
    generateMarketData();
    generateOrderBlocks();
    generateFVGs();
    
    setIsLoading(false);
    alert(`AI Analysis completed for ${selectedSymbol}! 
    
✅ Order blocks detected: ${orderBlocks.length}
✅ Fair Value Gaps identified: ${fvgs.length}
✅ Market structure analyzed
✅ Current price: ${currentPrice.toFixed(getPriceDecimals(selectedSymbol))}
✅ Price change: ${priceChange >= 0 ? '+' : ''}${priceChange.toFixed(2)}%
    
AI Confidence: ${87 + Math.floor(Math.random() * 10)}%`);
  };

  const getPriceDecimals = (symbol: string): number => {
    if (symbol.includes('JPY')) return 2;
    if (symbol.includes('.NS') || ['AAPL', 'TSLA', 'MSFT', 'GOOGL'].includes(symbol)) return 2;
    return 4; // For forex pairs
  };

  const generateOrderBlocks = () => {
    const blocks: OrderBlock[] = [
      {
        id: '1',
        price: 1.0823,
        type: 'bullish',
        strength: 85,
        timestamp: '2024-01-15 10:00:00',
      },
      {
        id: '2',
        price: 1.0847,
        type: 'bearish',
        strength: 72,
        timestamp: '2024-01-15 12:30:00',
      },
      {
        id: '3',
        price: 1.0801,
        type: 'bullish',
        strength: 91,
        timestamp: '2024-01-15 08:15:00',
      },
    ];
    setOrderBlocks(blocks);
  };

  const generateFVGs = () => {
    const gaps: FVG[] = [
      {
        id: '1',
        high: 1.0856,
        low: 1.0848,
        type: 'bearish',
        timestamp: '2024-01-15 11:00:00',
      },
      {
        id: '2',
        high: 1.0834,
        low: 1.0827,
        type: 'bullish',
        timestamp: '2024-01-15 09:30:00',
      },
    ];
    setFvgs(gaps);
  };

  const priceData = {
    labels: marketData.map(d => new Date(d.timestamp).toLocaleTimeString()),
    datasets: [
      {
        label: 'Price',
        data: marketData.map(d => d.close),
        borderColor: '#00ff88',
        backgroundColor: 'rgba(0, 255, 136, 0.1)',
        borderWidth: 2,
        fill: false,
      },
    ],
  };

  const chartOptions: ChartOptions<'line'> = {
    responsive: true,
    interaction: {
      mode: 'index' as const,
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: '#ffffff',
        },
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            return `Price: ${context.parsed.y.toFixed(4)}`;
          },
        },
      },
    },
    scales: {
      x: {
        ticks: { color: '#b0b0b0' },
        grid: { color: '#333' },
      },
      y: {
        ticks: { 
          color: '#b0b0b0',
          callback: function(value) {
            return (value as number).toFixed(4);
          },
        },
        grid: { color: '#333' },
      },
    },
  };

  const symbols = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
    'IRFC.NS', 'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFC.NS',
    'AAPL', 'TSLA', 'MSFT', 'GOOGL'
  ];
  const timeframes = ['15m', '30m', '1h', '4h', '1d'];

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 3 }}>
        Live Trading Analysis
      </Typography>

      {/* Symbol Selection and Current Price */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
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
        <Grid item xs={12} md={3}>
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
        <Grid item xs={12} md={6}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h6" color="text.secondary">
                {selectedSymbol}
              </Typography>
              <Typography variant="h4" component="div" sx={{ color: priceChange >= 0 ? '#00ff88' : '#ff5722' }}>
                {currentPrice.toFixed(getPriceDecimals(selectedSymbol))}
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                {priceChange >= 0 ? <TrendingUp /> : <TrendingDown />}
                <Typography 
                  variant="body1" 
                  sx={{ 
                    ml: 1, 
                    color: priceChange >= 0 ? '#00ff88' : '#ff5722',
                    fontWeight: 'bold'
                  }}
                >
                  {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
                </Typography>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Price Chart */}
        <Grid item xs={12} md={8}>
          <Paper className="trading-chart" sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Price Chart with ICT Analysis
            </Typography>
            <Line data={priceData} options={chartOptions} height={400} />
          </Paper>
        </Grid>

        {/* ICT Concepts Panel */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, backgroundColor: 'background.paper', height: 'fit-content' }}>
            <Typography variant="h6" gutterBottom>
              ICT Concepts Detected
            </Typography>
            
            {/* Order Blocks */}
            <Typography variant="subtitle1" sx={{ mt: 2, mb: 1, color: 'primary.main' }}>
              Order Blocks
            </Typography>
            <List dense>
              {orderBlocks.map((block) => (
                <ListItem key={block.id} sx={{ px: 0 }}>
                  <ListItemIcon>
                    <ShowChart color={block.type === 'bullish' ? 'primary' : 'secondary'} />
                  </ListItemIcon>
                  <ListItemText
                    primary={`${block.price.toFixed(4)} (${block.type})`}
                    secondary={`Strength: ${block.strength}%`}
                  />
                </ListItem>
              ))}
            </List>

            <Divider sx={{ my: 2 }} />

            {/* Fair Value Gaps */}
            <Typography variant="subtitle1" sx={{ mb: 1, color: 'secondary.main' }}>
              Fair Value Gaps
            </Typography>
            <List dense>
              {fvgs.map((fvg) => (
                <ListItem key={fvg.id} sx={{ px: 0 }}>
                  <ListItemIcon>
                    <Timeline color={fvg.type === 'bullish' ? 'primary' : 'secondary'} />
                  </ListItemIcon>
                  <ListItemText
                    primary={`${fvg.low.toFixed(4)} - ${fvg.high.toFixed(4)}`}
                    secondary={`${fvg.type} gap`}
                  />
                </ListItem>
              ))}
            </List>

            <Divider sx={{ my: 2 }} />

            {/* Market Structure */}
            <Typography variant="subtitle1" sx={{ mb: 1 }}>
              Market Structure
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              <Chip label="Higher High Confirmed" color="primary" size="small" />
              <Chip label="Bullish Structure" color="primary" size="small" />
              <Chip label="Trend: Upward" color="primary" size="small" />
            </Box>

            <Divider sx={{ my: 2 }} />

            {/* AI Recommendations */}
            <Typography variant="subtitle1" sx={{ mb: 1, color: 'primary.main' }}>
              AI Recommendations
            </Typography>
            <Box sx={{ p: 2, backgroundColor: 'rgba(0, 255, 136, 0.1)', borderRadius: 2 }}>
              <Typography variant="body2" sx={{ fontWeight: 'bold', color: 'primary.main' }}>
                BUY SIGNAL
              </Typography>
              <Typography variant="body2" sx={{ mt: 1 }}>
                Strong bullish order block at {orderBlocks[0]?.price.toFixed(4)}. 
                Entry recommended on retest.
              </Typography>
              <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
                Confidence: 87% | Risk: Medium
              </Typography>
            </Box>
          </Paper>
        </Grid>

        {/* Live Metrics */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, backgroundColor: 'background.paper' }}>
            <Typography variant="h6" gutterBottom>
              Live Market Metrics
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} md={2}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="primary.main">
                    72%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Bullish Sentiment
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={2}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="text.primary">
                    23
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Active Signals
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={2}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="secondary.main">
                    1.85
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Volatility Index
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={2}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="primary.main">
                    High
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Liquidity Level
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={2}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="text.primary">
                    4.2M
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Volume (1h)
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} md={2}>
                <Box sx={{ textAlign: 'center' }}>
                  <Typography variant="h4" color="primary.main">
                    Live
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Market Status
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>

        {/* Quick Actions */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, backgroundColor: 'background.paper' }}>
            <Typography variant="h6" gutterBottom>
              Quick Actions
            </Typography>
            <Grid container spacing={2}>
              <Grid item>
                <Button variant="contained" color="primary" startIcon={<TrendingUp />}>
                  Place Buy Order
                </Button>
              </Grid>
              <Grid item>
                <Button variant="contained" color="secondary" startIcon={<TrendingDown />}>
                  Place Sell Order
                </Button>
              </Grid>
              <Grid item>
                <Button 
                  variant="outlined" 
                  startIcon={<Psychology />} 
                  onClick={runAIAnalysis}
                  disabled={isLoading}
                >
                  {isLoading ? 'Analyzing...' : 'Run AI Analysis'}
                </Button>
              </Grid>
              <Grid item>
                <Button variant="outlined" startIcon={<Speed />}>
                  Auto-Trade Mode
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default TradingView;