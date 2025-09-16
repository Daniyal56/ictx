import React, { useState } from 'react';
import {
  Box,
  Grid,
  Paper,
  Typography,
  Card,
  CardContent,
  CardActions,
  Button,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Divider,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  ExpandMore,
  TrendingUp,
  Timeline,
  ShowChart,
  Assessment,
  Speed,
  Psychology,
  AutoAwesome,
  CheckCircle,
  Schedule,
} from '@mui/icons-material';

interface Strategy {
  id: string;
  name: string;
  description: string;
  performance: {
    winRate: number;
    totalReturn: number;
    maxDrawdown: number;
    trades: number;
  };
  status: 'active' | 'inactive' | 'paused';
  complexity: 'Beginner' | 'Intermediate' | 'Advanced';
  timeframe: string[];
  concepts: string[];
}

interface ICTConcept {
  id: string;
  name: string;
  category: string;
  description: string;
  implementation: 'Basic' | 'Advanced' | 'Expert';
  isImplemented: boolean;
}

const Strategies: React.FC = () => {
  const [strategies, setStrategies] = useState<Strategy[]>([
    {
      id: '1',
      name: 'Order Block Strategy',
      description: 'Identifies and trades institutional order blocks with high probability setups',
      performance: {
        winRate: 68.5,
        totalReturn: 45.67,
        maxDrawdown: -12.3,
        trades: 147,
      },
      status: 'active',
      complexity: 'Intermediate',
      timeframe: ['1h', '4h'],
      concepts: ['Order Blocks', 'Market Structure', 'Liquidity'],
    },
    {
      id: '2',
      name: 'Fair Value Gap Hunter',
      description: 'Exploits imbalances in price action using Fair Value Gap methodology',
      performance: {
        winRate: 72.1,
        totalReturn: 28.9,
        maxDrawdown: -8.7,
        trades: 98,
      },
      status: 'active',
      complexity: 'Advanced',
      timeframe: ['15m', '30m', '1h'],
      concepts: ['Fair Value Gaps', 'Imbalances', 'Premium/Discount'],
    },
    {
      id: '3',
      name: 'Market Structure Breaker',
      description: 'Trades market structure breaks with confirmation from multiple ICT concepts',
      performance: {
        winRate: 65.8,
        totalReturn: 32.45,
        maxDrawdown: -15.2,
        trades: 203,
      },
      status: 'active',
      complexity: 'Expert',
      timeframe: ['1h', '4h', '1d'],
      concepts: ['Market Structure', 'BOS', 'ChoCH', 'Swing Points'],
    },
    {
      id: '4',
      name: 'Liquidity Sweep Pro',
      description: 'Advanced liquidity hunting strategy targeting stops and key levels',
      performance: {
        winRate: 61.2,
        totalReturn: 41.8,
        maxDrawdown: -18.5,
        trades: 89,
      },
      status: 'paused',
      complexity: 'Expert',
      timeframe: ['30m', '1h'],
      concepts: ['Liquidity', 'Stop Hunts', 'Sweep Patterns'],
    },
    {
      id: '5',
      name: 'Institutional Candle Strategy',
      description: 'Leverages institutional candle patterns and rejection blocks',
      performance: {
        winRate: 74.3,
        totalReturn: 19.6,
        maxDrawdown: -6.8,
        trades: 76,
      },
      status: 'inactive',
      complexity: 'Beginner',
      timeframe: ['15m', '30m'],
      concepts: ['Institutional Candles', 'Rejection Blocks', 'Engulfing Patterns'],
    },
  ]);

  const [ictConcepts] = useState<ICTConcept[]>([
    {
      id: '1',
      name: 'Order Blocks',
      category: 'Core Concepts',
      description: 'Institutional order blocks representing areas of significant buying/selling interest',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '2',
      name: 'Fair Value Gaps (FVG)',
      category: 'Core Concepts',
      description: 'Price imbalances that represent inefficient price delivery',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '3',
      name: 'Market Structure',
      category: 'Core Concepts',
      description: 'Analysis of higher highs, higher lows, and structural patterns',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '4',
      name: 'Break of Structure (BOS)',
      category: 'Market Structure',
      description: 'Confirmation of trend continuation through structural breaks',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '5',
      name: 'Change of Character (ChoCH)',
      category: 'Market Structure',
      description: 'Early indication of potential trend reversal',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '6',
      name: 'Liquidity',
      category: 'Advanced Concepts',
      description: 'Identification and targeting of liquidity pools and stops',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '7',
      name: 'Premium & Discount',
      category: 'Advanced Concepts',
      description: 'Relative price positioning within ranges for optimal entries',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '8',
      name: 'Institutional Candles',
      category: 'Price Action',
      description: 'Specialized candlestick patterns indicating institutional activity',
      implementation: 'Basic',
      isImplemented: true,
    },
    {
      id: '9',
      name: 'Mitigation Blocks',
      category: 'Advanced Concepts',
      description: 'Areas where price returns to mitigate previous imbalances',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '10',
      name: 'Turtle Soup',
      category: 'Reversal Patterns',
      description: 'Reversal pattern targeting false breakouts and stop hunts',
      implementation: 'Basic',
      isImplemented: true,
    },
    // Add more concepts to reach 50+
    {
      id: '11',
      name: 'Displacement',
      category: 'Advanced Concepts',
      description: 'Rapid price movement indicating institutional involvement',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '12',
      name: 'Swing Points',
      category: 'Market Structure',
      description: 'Key pivot points used for structural analysis',
      implementation: 'Basic',
      isImplemented: true,
    },
    // ... Additional concepts would continue here
  ]);

  const toggleStrategyStatus = (id: string) => {
    setStrategies(prev => 
      prev.map(strategy => 
        strategy.id === id 
          ? { 
              ...strategy, 
              status: strategy.status === 'active' ? 'paused' : 'active' 
            }
          : strategy
      )
    );
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'primary';
      case 'paused': return 'warning';
      case 'inactive': return 'default';
      default: return 'default';
    }
  };

  const getComplexityColor = (complexity: string) => {
    switch (complexity) {
      case 'Beginner': return 'success';
      case 'Intermediate': return 'primary';
      case 'Advanced': return 'warning';
      case 'Expert': return 'error';
      default: return 'default';
    }
  };

  const getImplementationColor = (implementation: string) => {
    switch (implementation) {
      case 'Basic': return 'success';
      case 'Advanced': return 'primary';
      case 'Expert': return 'error';
      default: return 'default';
    }
  };

  const groupedConcepts = ictConcepts.reduce((acc, concept) => {
    if (!acc[concept.category]) {
      acc[concept.category] = [];
    }
    acc[concept.category].push(concept);
    return acc;
  }, {} as Record<string, ICTConcept[]>);

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" component="h1" gutterBottom sx={{ mb: 3 }}>
        Trading Strategies & ICT Concepts
      </Typography>

      <Grid container spacing={3}>
        {/* Active Strategies */}
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3, backgroundColor: 'background.paper', mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Active Trading Strategies
            </Typography>
            <Grid container spacing={2}>
              {strategies.map((strategy) => (
                <Grid item xs={12} key={strategy.id}>
                  <Card sx={{ backgroundColor: 'background.default' }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                        <Box>
                          <Typography variant="h6" gutterBottom>
                            {strategy.name}
                          </Typography>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            {strategy.description}
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', gap: 1, alignItems: 'center' }}>
                          <Chip 
                            label={strategy.status} 
                            color={getStatusColor(strategy.status) as any}
                            size="small"
                          />
                          <Chip 
                            label={strategy.complexity}
                            color={getComplexityColor(strategy.complexity) as any}
                            size="small"
                            variant="outlined"
                          />
                        </Box>
                      </Box>

                      <Grid container spacing={2} sx={{ mb: 2 }}>
                        <Grid item xs={3}>
                          <Typography variant="body2" color="text.secondary">Win Rate</Typography>
                          <Typography variant="h6" color="primary.main">
                            {strategy.performance.winRate}%
                          </Typography>
                        </Grid>
                        <Grid item xs={3}>
                          <Typography variant="body2" color="text.secondary">Total Return</Typography>
                          <Typography variant="h6" color="primary.main">
                            {strategy.performance.totalReturn}%
                          </Typography>
                        </Grid>
                        <Grid item xs={3}>
                          <Typography variant="body2" color="text.secondary">Max Drawdown</Typography>
                          <Typography variant="h6" color="error.main">
                            {strategy.performance.maxDrawdown}%
                          </Typography>
                        </Grid>
                        <Grid item xs={3}>
                          <Typography variant="body2" color="text.secondary">Trades</Typography>
                          <Typography variant="h6">
                            {strategy.performance.trades}
                          </Typography>
                        </Grid>
                      </Grid>

                      <Box sx={{ mb: 2 }}>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          Timeframes:
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          {strategy.timeframe.map((tf) => (
                            <Chip key={tf} label={tf} size="small" variant="outlined" />
                          ))}
                        </Box>
                      </Box>

                      <Box sx={{ mb: 2 }}>
                        <Typography variant="body2" color="text.secondary" gutterBottom>
                          ICT Concepts:
                        </Typography>
                        <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                          {strategy.concepts.map((concept) => (
                            <Chip key={concept} label={concept} size="small" />
                          ))}
                        </Box>
                      </Box>
                    </CardContent>
                    <CardActions>
                      <FormControlLabel
                        control={
                          <Switch 
                            checked={strategy.status === 'active'}
                            onChange={() => toggleStrategyStatus(strategy.id)}
                            color="primary"
                          />
                        }
                        label={strategy.status === 'active' ? 'Active' : 'Paused'}
                      />
                      <Button size="small" variant="outlined">
                        Configure
                      </Button>
                      <Button size="small" variant="outlined">
                        Backtest
                      </Button>
                    </CardActions>
                  </Card>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>

        {/* Strategy Performance Summary */}
        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3, backgroundColor: 'background.paper', mb: 3 }}>
            <Typography variant="h6" gutterBottom>
              Performance Overview
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Card sx={{ p: 2, backgroundColor: 'background.default' }}>
                <Typography variant="body2" color="text.secondary">Active Strategies</Typography>
                <Typography variant="h4" color="primary.main">
                  {strategies.filter(s => s.status === 'active').length}
                </Typography>
              </Card>
              <Card sx={{ p: 2, backgroundColor: 'background.default' }}>
                <Typography variant="body2" color="text.secondary">Avg Win Rate</Typography>
                <Typography variant="h4" color="primary.main">
                  {(strategies.reduce((acc, s) => acc + s.performance.winRate, 0) / strategies.length).toFixed(1)}%
                </Typography>
              </Card>
              <Card sx={{ p: 2, backgroundColor: 'background.default' }}>
                <Typography variant="body2" color="text.secondary">Total Trades</Typography>
                <Typography variant="h4">
                  {strategies.reduce((acc, s) => acc + s.performance.trades, 0)}
                </Typography>
              </Card>
            </Box>
          </Paper>

          {/* Quick Actions */}
          <Paper sx={{ p: 3, backgroundColor: 'background.paper' }}>
            <Typography variant="h6" gutterBottom>
              Quick Actions
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Button variant="contained" startIcon={<AutoAwesome />} fullWidth>
                Create New Strategy
              </Button>
              <Button variant="outlined" startIcon={<Assessment />} fullWidth>
                Performance Analysis
              </Button>
              <Button variant="outlined" startIcon={<Speed />} fullWidth>
                Auto-Optimize All
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* ICT Concepts Implementation */}
        <Grid item xs={12}>
          <Paper sx={{ p: 3, backgroundColor: 'background.paper' }}>
            <Typography variant="h6" gutterBottom>
              ICT Concepts Implementation Status ({ictConcepts.length}+ Concepts)
            </Typography>
            
            {Object.entries(groupedConcepts).map(([category, concepts]) => (
              <Accordion key={category} sx={{ mb: 1, backgroundColor: 'background.default' }}>
                <AccordionSummary expandIcon={<ExpandMore />}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                    <Typography variant="subtitle1">{category}</Typography>
                    <Chip 
                      label={`${concepts.length} concepts`} 
                      size="small" 
                      color="primary"
                    />
                    <Chip 
                      label={`${concepts.filter(c => c.isImplemented).length} implemented`}
                      size="small"
                      color="success"
                    />
                  </Box>
                </AccordionSummary>
                <AccordionDetails>
                  <Grid container spacing={2}>
                    {concepts.map((concept) => (
                      <Grid item xs={12} md={6} key={concept.id}>
                        <Card sx={{ p: 2, backgroundColor: 'background.paper' }}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                            {concept.isImplemented ? (
                              <CheckCircle color="success" fontSize="small" />
                            ) : (
                              <Schedule color="disabled" fontSize="small" />
                            )}
                            <Typography variant="subtitle2">{concept.name}</Typography>
                            <Chip 
                              label={concept.implementation}
                              size="small"
                              color={getImplementationColor(concept.implementation) as any}
                              variant="outlined"
                            />
                          </Box>
                          <Typography variant="body2" color="text.secondary">
                            {concept.description}
                          </Typography>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </AccordionDetails>
              </Accordion>
            ))}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Strategies;