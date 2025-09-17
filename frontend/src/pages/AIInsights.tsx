import React, { useState, useEffect } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  Button,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Avatar,
  LinearProgress,
  Divider,
  Alert,
  AlertTitle,
} from '@mui/material';
import {
  Psychology,
  TrendingUp,
  TrendingDown,
  AutoAwesome,
  Speed,
  Assessment,
  Lightbulb,
  Warning,
  CheckCircle,
  Info,
} from '@mui/icons-material';
import { Doughnut, Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
  ChartOptions,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

interface MarketSentiment {
  symbol: string;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  signals: string[];
  aiScore: number;
}

interface AIRecommendation {
  id: string;
  type: 'entry' | 'exit' | 'warning' | 'info';
  title: string;
  description: string;
  confidence: number;
  timestamp: string;
  symbol: string;
  priority: 'high' | 'medium' | 'low';
}

interface PatternRecognition {
  pattern: string;
  probability: number;
  timeframe: string;
  symbol: string;
  description: string;
}

const AIInsights: React.FC = () => {
  const [marketSentiment, setMarketSentiment] = useState<MarketSentiment[]>([
    {
      symbol: 'EURUSD',
      sentiment: 'bullish',
      confidence: 87,
      signals: ['Order Block Support', 'FVG Filled', 'Market Structure Intact'],
      aiScore: 8.7,
    },
    {
      symbol: 'GBPUSD',
      sentiment: 'bearish',
      confidence: 72,
      signals: ['Liquidity Swept', 'CHoCH Confirmed', 'Premium Rejection'],
      aiScore: 7.2,
    },
    {
      symbol: 'USDJPY',
      sentiment: 'neutral',
      confidence: 45,
      signals: ['Consolidation', 'Mixed Signals', 'Awaiting Catalyst'],
      aiScore: 4.5,
    },
    {
      symbol: 'AUDUSD',
      sentiment: 'bullish',
      confidence: 69,
      signals: ['BOS Confirmed', 'Discount Entry', 'Volume Spike'],
      aiScore: 6.9,
    },
  ]);

  const [recommendations, setRecommendations] = useState<AIRecommendation[]>([
    {
      id: '1',
      type: 'entry',
      title: 'EURUSD Buy Signal',
      description: 'Strong bullish order block at 1.0823 with high probability setup. Market structure supports continuation.',
      confidence: 87,
      timestamp: '2024-01-15 14:32:00',
      symbol: 'EURUSD',
      priority: 'high',
    },
    {
      id: '2',
      type: 'warning',
      title: 'GBPUSD Risk Alert',
      description: 'Potential liquidity sweep in progress. Exercise caution with long positions above 1.2650.',
      confidence: 72,
      timestamp: '2024-01-15 14:28:00',
      symbol: 'GBPUSD',
      priority: 'high',
    },
    {
      id: '3',
      type: 'info',
      title: 'Market Structure Update',
      description: 'USDJPY showing consolidation patterns. Breakout expected within next 4-8 hours.',
      confidence: 58,
      timestamp: '2024-01-15 14:15:00',
      symbol: 'USDJPY',
      priority: 'medium',
    },
    {
      id: '4',
      type: 'exit',
      title: 'Profit Taking Opportunity',
      description: 'AUDUSD approaching key resistance. Consider partial profit taking at current levels.',
      confidence: 74,
      timestamp: '2024-01-15 14:10:00',
      symbol: 'AUDUSD',
      priority: 'medium',
    },
  ]);

  const [patterns, setPatterns] = useState<PatternRecognition[]>([
    // Core ICT Concepts Patterns
    {
      pattern: 'Bullish Order Block',
      probability: 89,
      timeframe: '1H',
      symbol: 'EURUSD',
      description: 'Strong institutional support level identified',
    },
    {
      pattern: 'Bearish Order Block',
      probability: 84,
      timeframe: '4H',
      symbol: 'GBPUSD',
      description: 'Institutional resistance zone detected',
    },
    {
      pattern: 'Fair Value Gap (Bullish)',
      probability: 76,
      timeframe: '15M',
      symbol: 'USDJPY',
      description: 'Bullish imbalance detected, expecting fill',
    },
    {
      pattern: 'Fair Value Gap (Bearish)',
      probability: 73,
      timeframe: '30M',
      symbol: 'AUDUSD',
      description: 'Bearish imbalance identified, downside potential',
    },
    {
      pattern: 'Market Structure Break (BOS)',
      probability: 82,
      timeframe: '4H',
      symbol: 'USDCAD',
      description: 'Break of structure confirming trend continuation',
    },
    {
      pattern: 'Change of Character (ChoCH)',
      probability: 78,
      timeframe: '1H',
      symbol: 'NZDUSD',
      description: 'Market character change indicating potential reversal',
    },
    {
      pattern: 'Breaker Block Formation',
      probability: 85,
      timeframe: '1H',
      symbol: 'EURUSD',
      description: 'Failed order block creating new support/resistance',
    },
    {
      pattern: 'Liquidity Sweep Pattern',
      probability: 91,
      timeframe: '15M',
      symbol: 'GBPUSD',
      description: 'Stop hunt followed by reversal setup',
    },
    {
      pattern: 'Premium Discount Zone',
      probability: 74,
      timeframe: '4H',
      symbol: 'USDJPY',
      description: 'Price in premium zone, looking for reversal',
    },
    {
      pattern: 'Optimal Trade Entry (OTE)',
      probability: 88,
      timeframe: '1H',
      symbol: 'AUDUSD',
      description: '62-79% retracement zone identified for entry',
    },
    {
      pattern: 'Judas Swing (False Breakout)',
      probability: 79,
      timeframe: '30M',
      symbol: 'USDCAD',
      description: 'False breakout trapping retail traders',
    },
    {
      pattern: 'Liquidity Pool Target',
      probability: 83,
      timeframe: '15M',
      symbol: 'NZDUSD',
      description: 'Equal highs creating liquidity pool target',
    },
    {
      pattern: 'Power of 3 - Accumulation',
      probability: 86,
      timeframe: '4H',
      symbol: 'EURUSD',
      description: 'AMD model in accumulation phase',
    },
    {
      pattern: 'Power of 3 - Manipulation',
      probability: 81,
      timeframe: '1H',
      symbol: 'GBPUSD',
      description: 'Market manipulation phase identified',
    },
    {
      pattern: 'Power of 3 - Distribution',
      probability: 77,
      timeframe: '4H',
      symbol: 'USDJPY',
      description: 'Distribution phase - institutional selling',
    },
    {
      pattern: 'SMT Divergence',
      probability: 87,
      timeframe: '1H',
      symbol: 'AUDUSD',
      description: 'Smart money divergence across correlated pairs',
    },
    {
      pattern: 'Rejection Block',
      probability: 72,
      timeframe: '30M',
      symbol: 'USDCAD',
      description: 'Strong rejection with long wicks identified',
    },
    {
      pattern: 'Mitigation Block',
      probability: 84,
      timeframe: '1H',
      symbol: 'NZDUSD',
      description: 'Price returning to mitigate previous imbalance',
    },
    {
      pattern: 'Liquidity Void',
      probability: 75,
      timeframe: '15M',
      symbol: 'EURUSD',
      description: 'Price gap indicating liquidity void area',
    },
    {
      pattern: 'Supply Zone Formation',
      probability: 80,
      timeframe: '4H',
      symbol: 'GBPUSD',
      description: 'Fresh supply zone with institutional interest',
    },
    {
      pattern: 'Demand Zone Formation',
      probability: 82,
      timeframe: '1H',
      symbol: 'USDJPY',
      description: 'Strong demand zone with buying pressure',
    },
    {
      pattern: 'Killzone London Open',
      probability: 89,
      timeframe: '15M',
      symbol: 'EURUSD',
      description: 'London session killzone activation',
    },
    {
      pattern: 'Killzone New York Open',
      probability: 85,
      timeframe: '15M',
      symbol: 'USDCAD',
      description: 'New York session high-probability window',
    },
    {
      pattern: 'Session Liquidity Raid',
      probability: 78,
      timeframe: '30M',
      symbol: 'GBPUSD',
      description: 'Systematic liquidity raid during session',
    },
    {
      pattern: 'Weekly Opening Gap',
      probability: 73,
      timeframe: '1D',
      symbol: 'AUDUSD',
      description: 'Weekly opening gap indicating institutional bias',
    },
    {
      pattern: 'Daily Bias Confirmation',
      probability: 81,
      timeframe: '4H',
      symbol: 'NZDUSD',
      description: 'Daily directional bias confirmed by structure',
    },
    {
      pattern: 'Market Maker Algorithm',
      probability: 76,
      timeframe: '1H',
      symbol: 'USDJPY',
      description: 'Algorithmic price delivery pattern detected',
    },
    {
      pattern: 'Turtle Soup Reversal',
      probability: 74,
      timeframe: '30M',
      symbol: 'EURUSD',
      description: 'Failed breakout reversal setup confirmed',
    },
    {
      pattern: 'Fibonacci Confluence Zone',
      probability: 79,
      timeframe: '1H',
      symbol: 'GBPUSD',
      description: 'Multiple Fibonacci levels creating confluence',
    },
    {
      pattern: 'Time-Based Reversal',
      probability: 71,
      timeframe: '15M',
      symbol: 'USDCAD',
      description: 'Time-based analysis indicating reversal window',
    }
  ]);

  const sentimentData = {
    labels: ['Bullish', 'Bearish', 'Neutral'],
    datasets: [
      {
        data: [
          marketSentiment.filter(s => s.sentiment === 'bullish').length,
          marketSentiment.filter(s => s.sentiment === 'bearish').length,
          marketSentiment.filter(s => s.sentiment === 'neutral').length,
        ],
        backgroundColor: ['#00ff88', '#ff5722', '#757575'],
        borderColor: ['#00ff88', '#ff5722', '#757575'],
        borderWidth: 2,
      },
    ],
  };

  const aiPerformanceData = {
    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    datasets: [
      {
        label: 'AI Accuracy %',
        data: [78, 82, 76, 88, 84, 79, 86],
        borderColor: '#00ff88',
        backgroundColor: 'rgba(0, 255, 136, 0.1)',
        borderWidth: 2,
        fill: true,
      },
      {
        label: 'Signal Count',
        data: [23, 31, 19, 28, 34, 21, 27],
        borderColor: '#ff5722',
        backgroundColor: 'rgba(255, 87, 34, 0.1)',
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
        labels: { color: '#ffffff' },
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

  const doughnutOptions: ChartOptions<'doughnut'> = {
    responsive: true,
    plugins: {
      legend: {
        position: 'bottom' as const,
        labels: { color: '#ffffff' },
      },
    },
  };

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish': return <TrendingUp color="success" />;
      case 'bearish': return <TrendingDown color="error" />;
      default: return <TrendingUp color="disabled" />;
    }
  };

  const getSentimentColor = (sentiment: string): 'primary' | 'secondary' | 'error' | 'info' | 'success' | 'warning' => {
    switch (sentiment) {
      case 'bullish': return 'success';
      case 'bearish': return 'error';
      default: return 'primary';
    }
  };

  const getRecommendationIcon = (type: string) => {
    switch (type) {
      case 'entry': return <TrendingUp />;
      case 'exit': return <TrendingDown />;
      case 'warning': return <Warning />;
      default: return <Info />;
    }
  };

  const getRecommendationColor = (type: string) => {
    switch (type) {
      case 'entry': return 'success';
      case 'exit': return 'primary';
      case 'warning': return 'warning';
      default: return 'info';
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'high': return 'error';
      case 'medium': return 'warning';
      default: return 'success';
    }
  };

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 3 }}>
        AI Market Insights & Analysis
      </Typography>

      {/* AI Overview Cards */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card className="metric-card">
            <CardContent sx={{ textAlign: 'center' }}>
              <Psychology sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="h4" color="primary.main">
                AI Active
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Neural Network Status
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card className="metric-card">
            <CardContent sx={{ textAlign: 'center' }}>
              <AutoAwesome sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="h4" color="primary.main">
                84%
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Prediction Accuracy
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card className="metric-card">
            <CardContent sx={{ textAlign: 'center' }}>
              <Speed sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
              <Typography variant="h4" color="primary.main">
                156ms
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Analysis Speed
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card className="metric-card">
            <CardContent sx={{ textAlign: 'center' }}>
              <Assessment sx={{ fontSize: 40, color: "primary.main", mb: 1 }} />
              <Typography variant="h4" color="primary.main">
                {recommendations.length}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Active Signals
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Market Sentiment Analysis */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, backgroundColor: 'background.paper', mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Real-time Market Sentiment Analysis
            </Typography>
            <Grid container spacing={2}>
              {marketSentiment.map((sentiment) => (
                <Grid item xs={12} sm={6} key={sentiment.symbol}>
                  <Card sx={{ backgroundColor: 'background.default' }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <Avatar sx={{ mr: 2, backgroundColor: 'background.paper' }}>
                          {getSentimentIcon(sentiment.sentiment)}
                        </Avatar>
                        <Box sx={{ flexGrow: 1 }}>
                          <Typography variant="h6">{sentiment.symbol}</Typography>
                          <Chip 
                            label={sentiment.sentiment.toUpperCase()}
                            color={getSentimentColor(sentiment.sentiment)}
                            size="small"
                          />
                        </Box>
                        <Box sx={{ textAlign: 'right' }}>
                          <Typography variant="h6" color="primary.main">
                            {sentiment.aiScore}/10
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            AI Score
                          </Typography>
                        </Box>
                      </Box>
                      
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Confidence: {sentiment.confidence}%
                        </Typography>
                        <LinearProgress 
                          variant="determinate" 
                          value={sentiment.confidence} 
                          sx={{ height: 8, borderRadius: 4 }}
                          color={getSentimentColor(sentiment.sentiment)}
                        />
                      </Box>

                      <Box>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Key Signals:
                        </Typography>
                        {sentiment.signals.map((signal, index) => (
                          <Chip 
                            key={index}
                            label={signal}
                            size="small"
                            variant="outlined"
                            sx={{ mr: 0.5, mb: 0.5 }}
                          />
                        ))}
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>

          {/* AI Performance Chart */}
          <Paper sx={{ p: 3, backgroundColor: 'background.paper' }}>
            <Typography variant="h6" gutterBottom>
              AI Performance Analytics
            </Typography>
            <Line data={aiPerformanceData} options={chartOptions} />
          </Paper>
        </Grid>

        {/* Sentiment Distribution & Recommendations */}
        <Grid item xs={12} md={4}>
          {/* Sentiment Distribution */}
          <Paper sx={{ p: 3, backgroundColor: 'background.paper', mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Market Sentiment Distribution
            </Typography>
            <Doughnut data={sentimentData} options={doughnutOptions} />
          </Paper>

          {/* AI Recommendations */}
          <Paper sx={{ p: 3, backgroundColor: 'background.paper', mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Latest AI Recommendations
            </Typography>
            <Box sx={{ maxHeight: 400, overflow: 'auto' }} className="scrollbar">
              {recommendations.map((rec) => (
                <Alert 
                  key={rec.id}
                  severity={getRecommendationColor(rec.type) as any}
                  sx={{ mb: 2 }}
                  icon={getRecommendationIcon(rec.type)}
                >
                  <AlertTitle>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      {rec.title}
                      <Chip 
                        label={`${rec.confidence}%`}
                        size="small"
                        color={getPriorityColor(rec.priority) as any}
                      />
                    </Box>
                  </AlertTitle>
                  <Typography variant="body2" sx={{ mb: 1 }}>
                    {rec.description}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {rec.symbol} • {rec.timestamp}
                  </Typography>
                </Alert>
              ))}
            </Box>
          </Paper>
        </Grid>

        {/* Pattern Recognition */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, backgroundColor: 'background.paper' }}>
            <Typography variant="h6" gutterBottom>
              AI Pattern Recognition
            </Typography>
            <Grid container spacing={3}>
              {patterns.map((pattern, index) => (
                <Grid item xs={12} sm={6} md={3} key={index}>
                  <Card sx={{ backgroundColor: 'background.default', height: '100%' }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                        <Lightbulb sx={{ mr: 1, color: 'primary.main' }} />
                        <Typography variant="subtitle1" sx={{ fontWeight: 'bold' }}>
                          {pattern.pattern}
                        </Typography>
                      </Box>
                      
                      <Box sx={{ mb: 2 }}>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Probability: {pattern.probability}%
                        </Typography>
                        <LinearProgress 
                          variant="determinate" 
                          value={pattern.probability} 
                          sx={{ height: 6, borderRadius: 3 }}
                          color="primary"
                        />
                      </Box>

                      <Typography variant="body2" sx={{ mb: 2 }}>
                        {pattern.description}
                      </Typography>

                      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Chip label={pattern.symbol} size="small" variant="outlined" />
                        <Chip label={pattern.timeframe} size="small" color="primary" />
                      </Box>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>

        {/* AI Control Panel */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, backgroundColor: 'background.paper' }}>
            <Typography variant="h6" gutterBottom>
              AI Control Panel
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} md={3}>
                <Button 
                  variant="contained" 
                  fullWidth 
                  startIcon={<AutoAwesome />}
                  sx={{ mb: 1 }}
                >
                  Run Deep Analysis
                </Button>
              </Grid>
              <Grid item xs={12} md={3}>
                <Button 
                  variant="outlined" 
                  fullWidth 
                  startIcon={<Speed />}
                  sx={{ mb: 1 }}
                >
                  Optimize Models
                </Button>
              </Grid>
              <Grid item xs={12} md={3}>
                <Button 
                  variant="outlined" 
                  fullWidth 
                  startIcon={<Assessment />}
                  sx={{ mb: 1 }}
                >
                  Performance Report
                </Button>
              </Grid>
              <Grid item xs={12} md={3}>
                <Button 
                  variant="outlined" 
                  fullWidth 
                  startIcon={<Psychology />}
                  sx={{ mb: 1 }}
                >
                  Neural Network Status
                </Button>
              </Grid>
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AIInsights;