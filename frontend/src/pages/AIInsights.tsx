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
    {
      pattern: 'Bullish Order Block',
      probability: 89,
      timeframe: '1H',
      symbol: 'EURUSD',
      description: 'Strong institutional support level identified',
    },
    {
      pattern: 'Fair Value Gap',
      probability: 76,
      timeframe: '15M',
      symbol: 'GBPUSD',
      description: 'Imbalance detected, expecting fill',
    },
    {
      pattern: 'Market Structure Break',
      probability: 82,
      timeframe: '4H',
      symbol: 'USDJPY',
      description: 'Potential trend reversal signal',
    },
    {
      pattern: 'Liquidity Hunt',
      probability: 67,
      timeframe: '30M',
      symbol: 'AUDUSD',
      description: 'Stop hunt pattern developing',
    },
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