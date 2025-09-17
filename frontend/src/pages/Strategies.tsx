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
      complexity: 'Advanced',
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
      complexity: 'Advanced',
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
    // Core ICT Concepts (1-20)
    {
      id: '1',
      name: 'Market Structure (HH, HL, LH, LL)',
      category: 'Core Concepts',
      description: 'Analysis of higher highs, higher lows, lower highs, lower lows patterns',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '2',
      name: 'Liquidity (buy-side & sell-side)',
      category: 'Core Concepts',
      description: 'Identification of buy-side and sell-side liquidity areas',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '3',
      name: 'Liquidity Pools (equal highs/lows)',
      category: 'Core Concepts',
      description: 'Equal highs/lows and trendline liquidity identification',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '4',
      name: 'Order Blocks (Bullish & Bearish)',
      category: 'Core Concepts',
      description: 'Institutional order blocks representing areas of significant interest',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '5',
      name: 'Breaker Blocks',
      category: 'Core Concepts',
      description: 'Failed order blocks that become support/resistance reversal zones',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '6',
      name: 'Fair Value Gaps (FVG) / Imbalances',
      category: 'Core Concepts',
      description: 'Price imbalances representing inefficient price delivery',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '7',
      name: 'Rejection Blocks',
      category: 'Core Concepts',
      description: 'Areas where price shows strong rejection via wicks',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '8',
      name: 'Mitigation Blocks',
      category: 'Core Concepts',
      description: 'Areas where price returns to mitigate previous imbalances',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '9',
      name: 'Supply & Demand Zones',
      category: 'Core Concepts',
      description: 'High-volume zones indicating institutional supply/demand',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '10',
      name: 'Premium & Discount (OTE)',
      category: 'Core Concepts',
      description: 'Optimal Trade Entry zones using 62%-79% retracements',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '11',
      name: 'Dealing Ranges',
      category: 'Core Concepts',
      description: 'Consolidation ranges for buy/sell at discount/premium',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '12',
      name: 'Swing Highs & Swing Lows',
      category: 'Core Concepts',
      description: 'Key pivot points used for structural analysis',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '13',
      name: 'Market Maker Buy & Sell Models',
      category: 'Core Concepts',
      description: 'Institutional accumulation and distribution patterns',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '14',
      name: 'Market Maker Sell & Buy Programs',
      category: 'Core Concepts',
      description: 'Algorithmic selling and buying program identification',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '15',
      name: 'Judas Swing (false breakout)',
      category: 'Core Concepts',
      description: 'False breakouts at session opens designed to trap retail',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '16',
      name: 'Turtle Soup (stop-hunt strategy)',
      category: 'Core Concepts',
      description: 'Stop hunt reversal patterns targeting false breakouts',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '17',
      name: 'Power of 3 (AMD)',
      category: 'Core Concepts',
      description: 'Accumulation – Manipulation – Distribution cycle',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '18',
      name: 'Optimal Trade Entry (62%-79%)',
      category: 'Core Concepts',
      description: 'Precise entry zones using Fibonacci retracement levels',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '19',
      name: 'SMT Divergence',
      category: 'Core Concepts',
      description: 'Smart Money Divergence across correlated pairs',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '20',
      name: 'Liquidity Voids / Inefficiencies',
      category: 'Core Concepts',
      description: 'Price gaps and inefficient delivery areas',
      implementation: 'Advanced',
      isImplemented: true,
    },

    // Time & Price Theory (21-30)
    {
      id: '21',
      name: 'Killzones (London, NY, Asia)',
      category: 'Time & Price Theory',
      description: 'High-probability trading windows during major sessions',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '22',
      name: 'Midnight Open / Session Opens',
      category: 'Time & Price Theory',
      description: 'Key session opening times (00:00, 8:30, 9:30, 13:30)',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '23',
      name: 'Equilibrium & Fibonacci Ratios',
      category: 'Time & Price Theory',
      description: 'Key ratios: 50%, 62%, 70.5%, 79% for precise entries',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '24',
      name: 'Daily & Weekly Range Expectations',
      category: 'Time & Price Theory',
      description: 'Expected price movement ranges based on historical data',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '25',
      name: 'Session Liquidity Raids',
      category: 'Time & Price Theory',
      description: 'Liquidity sweeps during specific trading sessions',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '26',
      name: 'Weekly Profiles (WHLC)',
      category: 'Time & Price Theory',
      description: 'Weekly Open, High, Low, Close analysis',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '27',
      name: 'Daily Bias',
      category: 'Time & Price Theory',
      description: 'Direction bias using daily open and previous day levels',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '28',
      name: 'Weekly Bias',
      category: 'Time & Price Theory',
      description: 'Direction bias using weekly OHLC levels',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '29',
      name: 'Monthly Bias',
      category: 'Time & Price Theory',
      description: 'Long-term direction bias using monthly OHLC',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '30',
      name: 'Time of Day Highs & Lows',
      category: 'Time & Price Theory',
      description: 'AM/PM session separation and timing analysis',
      implementation: 'Advanced',
      isImplemented: true,
    },

    // Risk Management & Execution (31-39)
    {
      id: '31',
      name: 'Trade Journaling & Backtesting',
      category: 'Risk Management',
      description: 'Systematic recording and analysis of trading performance',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '32',
      name: 'Entry Models',
      category: 'Risk Management',
      description: 'FVG entry, OB entry, Breaker entry methodologies',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '33',
      name: 'Exit Models',
      category: 'Risk Management',
      description: 'Partial TP, full TP, scaling out strategies',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '34',
      name: 'Risk-to-Reward (RRR) Optimization',
      category: 'Risk Management',
      description: 'Optimizing risk-reward ratios for maximum profitability',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '35',
      name: 'Position Sizing',
      category: 'Risk Management',
      description: 'Calculating optimal position sizes based on risk tolerance',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '36',
      name: 'Drawdown Control',
      category: 'Risk Management',
      description: 'Managing and limiting account drawdown exposure',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '37',
      name: 'Compounding Models',
      category: 'Risk Management',
      description: 'Systematic approach to growing account equity',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '38',
      name: 'Daily Loss Limits',
      category: 'Risk Management',
      description: 'Maximum daily loss thresholds and controls',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '39',
      name: 'Probability Profiles (A+, B, C)',
      category: 'Risk Management',
      description: 'Classification of setups by probability of success',
      implementation: 'Advanced',
      isImplemented: true,
    },

    // Advanced Concepts (40-50)
    {
      id: '40',
      name: 'High Probability Trade Scenarios',
      category: 'Advanced Concepts',
      description: 'HTF bias + LTF confirmation strategies',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '41',
      name: 'Liquidity Runs',
      category: 'Advanced Concepts',
      description: 'Stop hunts, inducement, and fakeout identification',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '42',
      name: 'Reversals vs. Continuations',
      category: 'Advanced Concepts',
      description: 'Distinguishing between trend reversals and continuations',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '43',
      name: 'Accumulation & Distribution',
      category: 'Advanced Concepts',
      description: 'Institutional accumulation and distribution schematics',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '44',
      name: 'Order Flow',
      category: 'Advanced Concepts',
      description: 'Institutional narrative and order flow analysis',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '45',
      name: 'High/Low of the Day Identification',
      category: 'Advanced Concepts',
      description: 'Predicting and identifying daily extremes',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '46',
      name: 'Range Expansion',
      category: 'Advanced Concepts',
      description: 'Daily/weekly breakout and expansion patterns',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '47',
      name: 'Inside Day / Outside Day',
      category: 'Advanced Concepts',
      description: 'Daily range relationship analysis',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '48',
      name: 'Weekly Profiles Advanced',
      category: 'Advanced Concepts',
      description: 'Expansion, consolidation, reversal weekly patterns',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '49',
      name: 'IPDA Theory',
      category: 'Advanced Concepts',
      description: 'Interbank Price Delivery Algorithm theory',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '50',
      name: 'Algo-based Price Delivery',
      category: 'Advanced Concepts',
      description: 'ICT model of algorithmic market manipulation',
      implementation: 'Expert',
      isImplemented: true,
    },

    // Strategies / Playbooks (51-65)
    {
      id: '51',
      name: 'ICT Silver Bullet',
      category: 'Strategies',
      description: '15-minute window after NY Open high-probability setup',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '52',
      name: 'ICT Asian Range Breakout',
      category: 'Strategies',
      description: 'Trading breakouts from Asian session consolidation',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '53',
      name: 'ICT New York Reversal',
      category: 'Strategies',
      description: 'NY session reversal patterns and setups',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '54',
      name: 'ICT London Killzone',
      category: 'Strategies',
      description: 'London session high-probability trading strategy',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '55',
      name: 'ICT FVG Sniper Entry',
      category: 'Strategies',
      description: 'Precise Fair Value Gap entry methodology',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '56',
      name: 'ICT Order Block Strategy',
      category: 'Strategies',
      description: 'Comprehensive order block trading approach',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '57',
      name: 'ICT Breaker Block Strategy',
      category: 'Strategies',
      description: 'Failed order block reversal trading strategy',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '58',
      name: 'ICT Rejection Block Strategy',
      category: 'Strategies',
      description: 'Wick rejection and reversal trading methodology',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '59',
      name: 'ICT SMT Divergence Strategy',
      category: 'Strategies',
      description: 'Smart Money Divergence correlation trading',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '60',
      name: 'ICT Turtle Soup',
      category: 'Strategies',
      description: 'Liquidity raid reversal systematic approach',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '61',
      name: 'ICT Power of 3 Model',
      category: 'Strategies',
      description: 'Complete AMD cycle trading methodology',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '62',
      name: 'ICT Daily Bias + Liquidity Raid',
      category: 'Strategies',
      description: 'Combined daily bias and liquidity targeting',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '63',
      name: 'ICT AM Session Bias',
      category: 'Strategies',
      description: 'Morning session directional bias strategy',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '64',
      name: 'ICT PM Session Reversal',
      category: 'Strategies',
      description: 'Afternoon session reversal identification',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '65',
      name: 'ICT Optimal Trade Entry',
      category: 'Strategies',
      description: 'Refined 62%-79% retracement entry system',
      implementation: 'Expert',
      isImplemented: true,
    },
  ]);

  const [testResults, setTestResults] = useState<{[key: string]: any}>({});
  const [isTestingConcept, setIsTestingConcept] = useState<string | null>(null);

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

  const testConcept = async (conceptName: string) => {
    setIsTestingConcept(conceptName);
    
    try {
      // Convert concept name to strategy key
      const strategyKey = conceptName.toLowerCase()
        .replace(/\s+/g, '_')
        .replace(/[()&/-]/g, '')
        .replace(/__+/g, '_') + '_strategy';
      
      console.log(`Testing concept: ${conceptName} -> ${strategyKey}`);
      
      // Test the concept with real data
      const response = await fetch('http://localhost:8000/api/trading/test-strategy', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          strategy: strategyKey,
          symbol: 'EURUSD',
          timeframe: 'H1',
          test_periods: 100
        }),
      });
      
      if (response.ok) {
        const result = await response.json();
        setTestResults(prev => ({
          ...prev,
          [conceptName]: {
            success: true,
            setups: result.setups || [],
            timestamp: new Date().toISOString(),
            message: `Found ${result.setups?.length || 0} trading setups`
          }
        }));
        
        // Show success notification
        alert(`✅ ${conceptName} test successful!\nFound ${result.setups?.length || 0} trading setups\nLatest setup confidence: ${result.setups?.[0]?.confidence || 'N/A'}`);
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      console.error('Test failed:', error);
      setTestResults(prev => ({
        ...prev,
        [conceptName]: {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
          timestamp: new Date().toISOString()
        }
      }));
      
      // Show error notification
      alert(`❌ ${conceptName} test failed:\n${error instanceof Error ? error.message : 'Unknown error'}`);
    }
    
    setIsTestingConcept(null);
  };

  const viewImplementation = (conceptId: string) => {
    const concept = ictConcepts.find(c => c.id === conceptId);
    if (concept) {
      const strategyKey = concept.name.toLowerCase()
        .replace(/\s+/g, '_')
        .replace(/[()&/-]/g, '')
        .replace(/__+/g, '_') + '_strategy';
      
      // Open GitHub view of the implementation
      const githubUrl = `https://github.com/ajaygm18/ictx/blob/main/backend/strategies/ict_strategies.py#L${concept.id}`;
      
      alert(`📋 Implementation Details for ${concept.name}:

Strategy Function: ${strategyKey}
Implementation Level: ${concept.implementation}
Status: ${concept.isImplemented ? 'Implemented' : 'Pending'}
Category: ${concept.category}

The implementation can be found in:
backend/strategies/ict_strategies.py

GitHub URL: ${githubUrl}`);
    }
  };

  const runBacktest = async (conceptName: string) => {
    setIsTestingConcept(conceptName + '_backtest');
    
    try {
      const strategyKey = conceptName.toLowerCase()
        .replace(/\s+/g, '_')
        .replace(/[()&/-]/g, '')
        .replace(/__+/g, '_') + '_strategy';
      
      const response = await fetch('http://localhost:8000/api/backtesting/run', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          strategy: strategyKey,
          symbol: 'EURUSD',
          start_date: '2024-01-01',
          end_date: '2024-12-31',
          timeframe: 'H1',
          initial_capital: 10000
        }),
      });
      
      if (response.ok) {
        const result = await response.json();
        alert(`📊 Backtest Results for ${conceptName}:

Total Return: ${result.total_return || 'N/A'}%
Win Rate: ${result.win_rate || 'N/A'}%
Max Drawdown: ${result.max_drawdown || 'N/A'}%
Total Trades: ${result.total_trades || 'N/A'}
Profit Factor: ${result.profit_factor || 'N/A'}

Risk-Adjusted Return: ${result.sharpe_ratio || 'N/A'}`);
      } else {
        throw new Error('Backtest failed');
      }
    } catch (error) {
      alert(`❌ Backtest failed for ${conceptName}: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
    
    setIsTestingConcept(null);
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
              ICT Concepts Implementation Status ({ictConcepts.length} Concepts - Complete Implementation)
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
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                            {concept.description}
                          </Typography>
                          <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                            <Button
                              size="small"
                              variant="contained"
                              color="primary"
                              onClick={() => testConcept(concept.name)}
                              sx={{ fontSize: '0.7rem', minWidth: '70px' }}
                            >
                              Test Logic
                            </Button>
                            <Button
                              size="small"
                              variant="outlined"
                              color="secondary"
                              onClick={() => viewImplementation(concept.id)}
                              sx={{ fontSize: '0.7rem', minWidth: '70px' }}
                            >
                              View Code
                            </Button>
                            <Button
                              size="small"
                              variant="outlined"
                              color="info"
                              onClick={() => runBacktest(concept.name)}
                              sx={{ fontSize: '0.7rem', minWidth: '70px' }}
                            >
                              Backtest
                            </Button>
                          </Box>
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