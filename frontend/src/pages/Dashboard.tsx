import React, { useState, useEffect } from 'react';
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Paper,
  Button,
  Chip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  AccountBalance,
  ShowChart,
  Assessment,
  Psychology,
} from '@mui/icons-material';
import { Line } from 'react-chartjs-2';
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

interface DashboardStats {
  totalProfit: number;
  winRate: number;
  totalTrades: number;
  activeStrategies: number;
  portfolioValue: number;
  dailyPnL: number;
}

interface RecentTrade {
  id: string;
  symbol: string;
  type: 'BUY' | 'SELL';
  price: number;
  quantity: number;
  profit: number;
  timestamp: string;
  strategy: string;
}

const Dashboard: React.FC = () => {
  const [stats, setStats] = useState<DashboardStats>({
    totalProfit: 15420.50,
    winRate: 68.5,
    totalTrades: 147,
    activeStrategies: 8,
    portfolioValue: 125340.75,
    dailyPnL: 2847.33,
  });

  const [recentTrades, setRecentTrades] = useState<RecentTrade[]>([
    {
      id: '1',
      symbol: 'EURUSD',
      type: 'BUY',
      price: 1.0835,
      quantity: 10000,
      profit: 145.80,
      timestamp: '2024-01-15 14:32:11',
      strategy: 'Order Block',
    },
    {
      id: '2',
      symbol: 'GBPUSD',
      type: 'SELL',
      price: 1.2654,
      quantity: 8000,
      profit: -87.50,
      timestamp: '2024-01-15 13:45:22',
      strategy: 'Fair Value Gap',
    },
    {
      id: '3',
      symbol: 'USDJPY',
      type: 'BUY',
      price: 147.85,
      quantity: 5000,
      profit: 234.15,
      timestamp: '2024-01-15 12:18:45',
      strategy: 'Market Structure',
    },
  ]);

  const portfolioData = {
    labels: ['Jan 1', 'Jan 3', 'Jan 5', 'Jan 7', 'Jan 9', 'Jan 11', 'Jan 13', 'Jan 15'],
    datasets: [
      {
        label: 'Portfolio Value',
        data: [100000, 102500, 101800, 105200, 108300, 106900, 112450, 125340.75],
        borderColor: '#00ff88',
        backgroundColor: 'rgba(0, 255, 136, 0.1)',
        borderWidth: 2,
        fill: true,
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
      title: {
        display: false,
      },
    },
    scales: {
      x: {
        ticks: {
          color: '#b0b0b0',
        },
        grid: {
          color: '#333',
        },
      },
      y: {
        ticks: {
          color: '#b0b0b0',
        },
        grid: {
          color: '#333',
        },
      },
    },
  };

  const StatCard: React.FC<{
    title: string;
    value: string | number;
    icon: React.ReactNode;
    color?: string;
    subtitle?: string;
  }> = ({ title, value, icon, color = '#00ff88', subtitle }) => (
    <Card className="metric-card">
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
          <Box sx={{ color: color, mr: 1 }}>{icon}</Box>
          <Typography variant="h6" component="div">
            {title}
          </Typography>
        </Box>
        <Typography variant="h4" component="div" sx={{ color: color, fontWeight: 'bold' }}>
          {value}
        </Typography>
        {subtitle && (
          <Typography variant="body2" color="text.secondary">
            {subtitle}
          </Typography>
        )}
      </CardContent>
    </Card>
  );

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 3 }}>
        Trading Dashboard
      </Typography>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} sm={6} md={2}>
          <StatCard
            title="Portfolio"
            value={`$${stats.portfolioValue.toLocaleString()}`}
            icon={<AccountBalance />}
            subtitle="Total Value"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <StatCard
            title="Daily P&L"
            value={`$${stats.dailyPnL.toFixed(2)}`}
            icon={<TrendingUp />}
            color={stats.dailyPnL >= 0 ? '#00ff88' : '#ff5722'}
            subtitle="Today"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <StatCard
            title="Total Profit"
            value={`$${stats.totalProfit.toFixed(2)}`}
            icon={<ShowChart />}
            subtitle="All Time"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <StatCard
            title="Win Rate"
            value={`${stats.winRate}%`}
            icon={<Assessment />}
            subtitle="Success Rate"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <StatCard
            title="Total Trades"
            value={stats.totalTrades}
            icon={<TrendingUp />}
            subtitle="Executed"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={2}>
          <StatCard
            title="Strategies"
            value={stats.activeStrategies}
            icon={<Psychology />}
            subtitle="Active"
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Portfolio Chart */}
        <Grid item xs={12} md={8}>
          <Paper className="trading-chart" sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Portfolio Performance
            </Typography>
            <Line data={portfolioData} options={chartOptions} />
          </Paper>
        </Grid>

        {/* Recent Trades */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, backgroundColor: 'background.paper' }}>
            <Typography variant="h6" gutterBottom>
              Recent Trades
            </Typography>
            <Box sx={{ maxHeight: 400, overflow: 'auto' }} className="scrollbar">
              {recentTrades.map((trade) => (
                <Box
                  key={trade.id}
                  sx={{
                    p: 2,
                    mb: 2,
                    backgroundColor: 'background.default',
                    borderRadius: 2,
                    border: '1px solid #333',
                  }}
                >
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                    <Typography variant="body1" sx={{ fontWeight: 'bold' }}>
                      {trade.symbol}
                    </Typography>
                    <Chip
                      label={trade.type}
                      size="small"
                      color={trade.type === 'BUY' ? 'primary' : 'secondary'}
                    />
                  </Box>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    {trade.strategy}
                  </Typography>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                    <Typography variant="body2">
                      ${trade.price} × {trade.quantity.toLocaleString()}
                    </Typography>
                    <Typography
                      variant="body2"
                      className={trade.profit >= 0 ? 'profit' : 'loss'}
                      sx={{ fontWeight: 'bold' }}
                    >
                      ${trade.profit.toFixed(2)}
                    </Typography>
                  </Box>
                  <Typography variant="caption" color="text.secondary">
                    {trade.timestamp}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Paper>
        </Grid>

        {/* AI Recommendations */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, backgroundColor: 'background.paper' }}>
            <Typography variant="h6" gutterBottom>
              AI Market Analysis & Recommendations
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={4}>
                <Box sx={{ p: 2, backgroundColor: 'rgba(0, 255, 136, 0.1)', borderRadius: 2 }}>
                  <Typography variant="subtitle1" sx={{ color: 'primary.main', fontWeight: 'bold' }}>
                    EURUSD - BUY SIGNAL
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Strong order block detected at 1.0820. Market structure showing bullish continuation pattern.
                  </Typography>
                  <Chip label="Confidence: 87%" size="small" sx={{ mt: 1 }} color="primary" />
                </Box>
              </Grid>
              <Grid item xs={12} md={4}>
                <Box sx={{ p: 2, backgroundColor: 'rgba(255, 87, 34, 0.1)', borderRadius: 2 }}>
                  <Typography variant="subtitle1" sx={{ color: 'secondary.main', fontWeight: 'bold' }}>
                    GBPUSD - SELL SIGNAL
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Fair Value Gap identified. Expecting bearish retracement to liquidity zone.
                  </Typography>
                  <Chip label="Confidence: 72%" size="small" sx={{ mt: 1 }} color="secondary" />
                </Box>
              </Grid>
              <Grid item xs={12} md={4}>
                <Box sx={{ p: 2, backgroundColor: 'rgba(255, 255, 255, 0.05)', borderRadius: 2 }}>
                  <Typography variant="subtitle1" sx={{ color: 'text.primary', fontWeight: 'bold' }}>
                    USDJPY - NEUTRAL
                  </Typography>
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Consolidating within range. Awaiting break of key levels for directional bias.
                  </Typography>
                  <Chip label="Confidence: 45%" size="small" sx={{ mt: 1 }} />
                </Box>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Dashboard;