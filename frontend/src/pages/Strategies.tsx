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
    // Core ICT Strategies (1-20)
    {
      id: '1',
      name: 'Market Structure Strategy',
      description: 'Analyzes higher highs, higher lows, lower highs, lower lows patterns',
      performance: { winRate: 68.5, totalReturn: 45.67, maxDrawdown: -12.3, trades: 147 },
      status: 'active',
      complexity: 'Intermediate',
      timeframe: ['1h', '4h'],
      concepts: ['Market Structure', 'HH', 'HL', 'LH', 'LL'],
    },
    {
      id: '2',
      name: 'Liquidity Strategy',
      description: 'Identifies and trades buy-side and sell-side liquidity areas',
      performance: { winRate: 72.1, totalReturn: 28.9, maxDrawdown: -8.7, trades: 98 },
      status: 'active',
      complexity: 'Advanced',
      timeframe: ['15m', '30m', '1h'],
      concepts: ['Liquidity', 'Buy-side', 'Sell-side'],
    },
    {
      id: '3',
      name: 'Order Block Strategy',
      description: 'Trades institutional order blocks with high probability setups',
      performance: { winRate: 75.3, totalReturn: 52.4, maxDrawdown: -10.1, trades: 203 },
      status: 'active',
      complexity: 'Advanced',
      timeframe: ['1h', '4h'],
      concepts: ['Order Blocks', 'Bullish OB', 'Bearish OB'],
    },
    {
      id: '4',
      name: 'Breaker Block Strategy',
      description: 'Failed order blocks that become support/resistance reversal zones',
      performance: { winRate: 61.2, totalReturn: 41.8, maxDrawdown: -18.5, trades: 89 },
      status: 'active',
      complexity: 'Advanced',
      timeframe: ['30m', '1h'],
      concepts: ['Breaker Blocks', 'Failed OB', 'Reversal Zones'],
    },
    {
      id: '5',
      name: 'Fair Value Gap Strategy',
      description: 'Exploits price imbalances and inefficient price delivery',
      performance: { winRate: 74.3, totalReturn: 19.6, maxDrawdown: -6.8, trades: 76 },
      status: 'active',
      complexity: 'Intermediate',
      timeframe: ['15m', '30m'],
      concepts: ['Fair Value Gaps', 'Imbalances', 'Inefficiency'],
    },
    {
      id: '6',
      name: 'Rejection Block Strategy',
      description: 'Areas where price shows strong rejection via wicks',
      performance: { winRate: 69.8, totalReturn: 34.2, maxDrawdown: -14.3, trades: 112 },
      status: 'active',
      complexity: 'Beginner',
      timeframe: ['1h', '4h'],
      concepts: ['Rejection Blocks', 'Wicks', 'Price Rejection'],
    },
    {
      id: '7',
      name: 'Premium Discount Strategy',
      description: 'Trades based on premium and discount pricing zones',
      performance: { winRate: 65.7, totalReturn: 38.9, maxDrawdown: -16.2, trades: 156 },
      status: 'active',
      complexity: 'Intermediate',
      timeframe: ['1h', '4h'],
      concepts: ['Premium', 'Discount', 'OTE'],
    },
    {
      id: '8',
      name: 'Liquidity Pools Strategy',
      description: 'Equal highs/lows and trendline liquidity identification',
      performance: { winRate: 71.4, totalReturn: 43.6, maxDrawdown: -11.8, trades: 94 },
      status: 'active',
      complexity: 'Advanced',
      timeframe: ['30m', '1h', '4h'],
      concepts: ['Liquidity Pools', 'Equal Highs', 'Equal Lows'],
    },
    {
      id: '9',
      name: 'Mitigation Block Strategy',
      description: 'Blocks that mitigate previous moves and provide entry opportunities',
      performance: { winRate: 67.2, totalReturn: 29.3, maxDrawdown: -13.7, trades: 123 },
      status: 'active',
      complexity: 'Advanced',
      timeframe: ['15m', '30m', '1h'],
      concepts: ['Mitigation Blocks', 'Mitigation', 'Entry Points'],
    },
    {
      id: '10',
      name: 'Supply Demand Zones Strategy',
      description: 'Traditional supply and demand zone analysis with ICT principles',
      performance: { winRate: 62.8, totalReturn: 47.1, maxDrawdown: -19.4, trades: 87 },
      status: 'active',
      complexity: 'Intermediate',
      timeframe: ['1h', '4h', '1d'],
      concepts: ['Supply Zones', 'Demand Zones', 'S&D'],
    },
    
    // Time & Price Theory Strategies (11-20)
    {
      id: '11',
      name: 'Killzones Strategy',
      description: 'Trades during high-probability time windows (London, NY, Asia)',
      performance: { winRate: 76.9, totalReturn: 55.8, maxDrawdown: -9.2, trades: 167 },
      status: 'active',
      complexity: 'Advanced',
      timeframe: ['15m', '30m'],
      concepts: ['Killzones', 'London Session', 'NY Session'],
    },
    {
      id: '12',
      name: 'Session Opens Strategy',
      description: 'Trades session opens and midnight open patterns',
      performance: { winRate: 68.4, totalReturn: 31.7, maxDrawdown: -12.6, trades: 134 },
      status: 'active',
      complexity: 'Intermediate',
      timeframe: ['1h', '4h'],
      concepts: ['Session Opens', 'Midnight Open', 'Time Analysis'],
    },
    {
      id: '13',
      name: 'Fibonacci Ratios Strategy',
      description: 'Uses Fibonacci retracements and extensions for entries',
      performance: { winRate: 64.1, totalReturn: 26.9, maxDrawdown: -15.8, trades: 98 },
      status: 'active',
      complexity: 'Intermediate',
      timeframe: ['1h', '4h'],
      concepts: ['Fibonacci', 'Retracements', 'Extensions'],
    },
    {
      id: '14',
      name: 'Power of Three Strategy',
      description: 'Accumulation, Manipulation, Distribution model',
      performance: { winRate: 73.6, totalReturn: 49.2, maxDrawdown: -8.4, trades: 156 },
      status: 'active',
      complexity: 'Advanced',
      timeframe: ['30m', '1h', '4h'],
      concepts: ['Power of 3', 'AMD Model', 'Market Phases'],
    },
    {
      id: '15',
      name: 'SMT Divergence Strategy',
      description: 'Smart Money Theory divergence between correlated pairs',
      performance: { winRate: 70.2, totalReturn: 44.3, maxDrawdown: -13.1, trades: 78 },
      status: 'active',
      complexity: 'Advanced',
      timeframe: ['1h', '4h'],
      concepts: ['SMT', 'Divergence', 'Correlation'],
    },
    
    // Advanced Strategies (16-25)
    {
      id: '16',
      name: 'Optimal Trade Entry Strategy',
      description: 'Finds optimal entry points using multiple ICT concepts',
      performance: { winRate: 78.5, totalReturn: 62.7, maxDrawdown: -7.3, trades: 142 },
      status: 'active',
      complexity: 'Advanced',
      timeframe: ['15m', '30m', '1h'],
      concepts: ['OTE', 'Optimal Entry', 'Multi-Concept'],
    },
    {
      id: '17',
      name: 'Judas Swing Strategy',
      description: 'False breakout patterns that trap retail traders',
      performance: { winRate: 66.7, totalReturn: 37.8, maxDrawdown: -14.9, trades: 89 },
      status: 'active',
      complexity: 'Advanced',
      timeframe: ['30m', '1h'],
      concepts: ['Judas Swing', 'False Breakout', 'Trap'],
    },
    {
      id: '18',
      name: 'Turtle Soup Strategy',
      description: 'Reversal strategy based on failed breakouts',
      performance: { winRate: 59.3, totalReturn: 33.1, maxDrawdown: -17.6, trades: 67 },
      status: 'paused',
      complexity: 'Advanced',
      timeframe: ['1h', '4h'],
      concepts: ['Turtle Soup', 'Failed Breakout', 'Reversal'],
    },
    {
      id: '19',
      name: 'Liquidity Voids Strategy',
      description: 'Trades areas where liquidity has been removed',
      performance: { winRate: 71.8, totalReturn: 46.5, maxDrawdown: -11.2, trades: 123 },
      status: 'active',
      complexity: 'Advanced',
      timeframe: ['30m', '1h', '4h'],
      concepts: ['Liquidity Voids', 'Void Areas', 'Liquidity Removal'],
    },
    {
      id: '20',
      name: 'Market Maker Models Strategy',
      description: 'Understanding institutional market making algorithms',
      performance: { winRate: 74.2, totalReturn: 51.9, maxDrawdown: -9.8, trades: 167 },
      status: 'active',
      complexity: 'Advanced',
      timeframe: ['1h', '4h'],
      concepts: ['Market Makers', 'Algorithms', 'Institutional Flow'],
    },
    
    // Additional Active Strategies (21-30)
    {
      id: '21',
      name: 'Range Expectations Strategy',
      description: 'Daily and weekly range projections and trading',
      performance: { winRate: 63.4, totalReturn: 28.7, maxDrawdown: -16.3, trades: 145 },
      status: 'active',
      complexity: 'Intermediate',
      timeframe: ['4h', '1d'],
      concepts: ['Daily Range', 'Weekly Range', 'Expectations'],
    },
    {
      id: '22',
      name: 'Session Liquidity Raids Strategy',
      description: 'Targets liquidity raids during specific sessions',
      performance: { winRate: 69.1, totalReturn: 42.6, maxDrawdown: -12.8, trades: 98 },
      status: 'active',
      complexity: 'Advanced',
      timeframe: ['15m', '30m', '1h'],
      concepts: ['Liquidity Raids', 'Session Analysis', 'Raids'],
    },
    {
      id: '23',
      name: 'Weekly Profiles Strategy',
      description: 'Weekly market profile and bias analysis',
      performance: { winRate: 67.8, totalReturn: 35.4, maxDrawdown: -14.1, trades: 87 },
      status: 'active',
      complexity: 'Intermediate',
      timeframe: ['4h', '1d'],
      concepts: ['Weekly Profiles', 'Market Profile', 'Weekly Bias'],
    },
    {
      id: '24',
      name: 'Daily Bias Strategy',
      description: 'Daily directional bias based on ICT methodology',
      performance: { winRate: 72.5, totalReturn: 48.3, maxDrawdown: -10.7, trades: 234 },
      status: 'active',
      complexity: 'Intermediate',
      timeframe: ['1h', '4h'],
      concepts: ['Daily Bias', 'Directional Bias', 'Daily Analysis'],
    },
    {
      id: '25',
      name: 'Monthly Bias Strategy',
      description: 'Long-term monthly bias and trend analysis',
      performance: { winRate: 58.9, totalReturn: 67.2, maxDrawdown: -22.4, trades: 45 },
      status: 'active',
      complexity: 'Advanced',
      timeframe: ['1d', '1w'],
      concepts: ['Monthly Bias', 'Long-term Trends', 'Monthly Analysis'],
    }
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
    
    // Time & Price Theory (21-40)
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
      description: 'Key retracement levels and equilibrium pricing',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '24',
      name: 'Daily & Weekly Range Expectations',
      category: 'Time & Price Theory',
      description: 'Projected daily and weekly trading ranges',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '25',
      name: 'Session Liquidity Raids',
      category: 'Time & Price Theory',
      description: 'Systematic raids on liquidity during sessions',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '26',
      name: 'Weekly Profiles & Opening Range',
      category: 'Time & Price Theory',
      description: 'Weekly opening range and profile analysis',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '27',
      name: 'Daily Bias Confirmation',
      category: 'Time & Price Theory',
      description: 'Daily directional bias confirmation methods',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '28',
      name: 'Weekly Bias Assessment',
      category: 'Time & Price Theory',
      description: 'Weekly trend bias and direction analysis',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '29',
      name: 'Monthly Bias & Trend',
      category: 'Time & Price Theory',
      description: 'Long-term monthly bias and trend assessment',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '30',
      name: 'Time of Day Analysis',
      category: 'Time & Price Theory',
      description: 'Specific hour-by-hour market behavior patterns',
      implementation: 'Advanced',
      isImplemented: true,
    },
    
    // Advanced Concepts (31-50)
    {
      id: '31',
      name: 'High Probability Trading Scenarios',
      category: 'Advanced Concepts',
      description: 'Multiple confluence scenarios for high probability trades',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '32',
      name: 'Liquidity Runs & Sweeps',
      category: 'Advanced Concepts',
      description: 'Stop runs and liquidity sweep patterns',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '33',
      name: 'Reversals vs Continuations',
      category: 'Advanced Concepts',
      description: 'Distinguishing between reversal and continuation patterns',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '34',
      name: 'Accumulation & Distribution',
      category: 'Advanced Concepts',
      description: 'Institutional accumulation and distribution identification',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '35',
      name: 'Order Flow Analysis',
      category: 'Advanced Concepts',
      description: 'Reading institutional order flow and positioning',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '36',
      name: 'Institutional Sponsorship',
      category: 'Advanced Concepts',
      description: 'Identifying when institutions are backing a move',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '37',
      name: 'Market Efficiency Theory',
      category: 'Advanced Concepts',
      description: 'Price efficiency and inefficiency identification',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '38',
      name: 'Fractal Repetition',
      category: 'Advanced Concepts',
      description: 'Pattern repetition across multiple timeframes',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '39',
      name: 'Intermarket Analysis',
      category: 'Advanced Concepts',
      description: 'Cross-market correlations and relationships',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '40',
      name: 'Seasonal Tendencies',
      category: 'Advanced Concepts',
      description: 'Monthly and quarterly seasonal market patterns',
      implementation: 'Advanced',
      isImplemented: true,
    },
    
    // Risk Management & Execution (41-60)
    {
      id: '41',
      name: 'Trade Journaling & Review',
      category: 'Risk Management',
      description: 'Systematic trade documentation and analysis',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '42',
      name: 'Entry Models & Techniques',
      category: 'Risk Management',
      description: 'Precise entry methods and confirmation techniques',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '43',
      name: 'Exit Models & Management',
      category: 'Risk Management',
      description: 'Position exit strategies and trade management',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '44',
      name: 'Risk-to-Reward Ratios',
      category: 'Risk Management',
      description: 'Optimal risk-reward ratio calculations and management',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '45',
      name: 'Position Sizing Models',
      category: 'Risk Management',
      description: 'Dynamic position sizing based on account and risk',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '46',
      name: 'Drawdown Control Systems',
      category: 'Risk Management',
      description: 'Maximum drawdown limits and recovery strategies',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '47',
      name: 'Compounding Models',
      category: 'Risk Management',
      description: 'Account growth and compounding strategies',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '48',
      name: 'Daily Loss Limits',
      category: 'Risk Management',
      description: 'Daily maximum loss thresholds and stops',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '49',
      name: 'Probability Profiles',
      category: 'Risk Management',
      description: 'Trade probability assessment and profiling',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '50',
      name: 'Psychological Framework',
      category: 'Risk Management',
      description: 'Trading psychology and emotional control systems',
      implementation: 'Advanced',
      isImplemented: true,
    },
    
    // Additional ICT Concepts (51-65+)
    {
      id: '51',
      name: 'New Day Opening Gap',
      category: 'Advanced Patterns',
      description: 'Gap analysis at new day opens and their implications',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '52',
      name: 'Opening Range Breakouts',
      category: 'Advanced Patterns',
      description: 'Trading session opening range breakout patterns',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '53',
      name: 'Institutional Reference Points',
      category: 'Advanced Patterns',
      description: 'Key levels institutions use for decision making',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '54',
      name: 'Algorithmic Price Delivery',
      category: 'Advanced Patterns',
      description: 'How algorithms deliver price to key levels',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '55',
      name: 'Market Profile Integration',
      category: 'Advanced Patterns',
      description: 'Market profile concepts within ICT methodology',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '56',
      name: 'Volume Spread Analysis',
      category: 'Advanced Patterns',
      description: 'Price-volume relationship analysis',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '57',
      name: 'Currency Strength Analysis',
      category: 'Advanced Patterns',
      description: 'Individual currency strength and weakness measurement',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '58',
      name: 'News Trading Methodology',
      category: 'Advanced Patterns',
      description: 'Trading around high-impact news events',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '59',
      name: 'Multi-Timeframe Analysis',
      category: 'Advanced Patterns',
      description: 'Comprehensive analysis across multiple timeframes',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '60',
      name: 'Market Regime Identification',
      category: 'Advanced Patterns',
      description: 'Trending vs ranging market identification',
      implementation: 'Advanced',
      isImplemented: true,
    },
    {
      id: '61',
      name: 'Smart Money Concepts',
      category: 'Advanced Patterns',
      description: 'Advanced smart money theory concepts',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '62',
      name: 'Institutional Level Trading',
      category: 'Advanced Patterns',
      description: 'Trading at institutional decision levels',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '63',
      name: 'Market Structure Shifts',
      category: 'Advanced Patterns',
      description: 'Identifying major market structure changes',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '64',
      name: 'Time-Based Analysis',
      category: 'Advanced Patterns',
      description: 'Advanced time-based trading techniques',
      implementation: 'Expert',
      isImplemented: true,
    },
    {
      id: '65',
      name: 'ICT Mentorship Concepts',
      category: 'Advanced Patterns',
      description: 'Advanced concepts from ICT mentorship programs',
      implementation: 'Expert',
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
      // Map concept names to actual strategy keys in backend
      const strategyMap: {[key: string]: string} = {
        'Market Structure (HH, HL, LH, LL)': 'market_structure_strategy',
        'Liquidity (buy-side & sell-side)': 'liquidity_strategy',
        'Liquidity Pools (equal highs/lows)': 'liquidity_pools_strategy',
        'Order Blocks (Bullish & Bearish)': 'order_block_strategy',
        'Breaker Blocks': 'breaker_block_strategy',
        'Fair Value Gaps (FVG) / Imbalances': 'fair_value_gap_strategy',
        'Rejection Blocks': 'rejection_block_strategy',
        'Mitigation Blocks': 'mitigation_block_strategy',
        'Supply & Demand Zones': 'supply_demand_zones_strategy',
        'Premium & Discount (OTE)': 'premium_discount_strategy',
        'Dealing Ranges': 'dealing_ranges_strategy',
        'Swing Highs & Swing Lows': 'swing_points_strategy',
        'Market Maker Buy & Sell Models': 'market_maker_models_strategy',
        'Market Maker Sell & Buy Programs': 'market_maker_models_strategy',
        'Judas Swing (false breakout)': 'judas_swing_strategy',
        'Turtle Soup (stop-hunt strategy)': 'turtle_soup_strategy',
        'Power of 3 (AMD)': 'power_of_three_strategy',
        'Optimal Trade Entry (62%-79%)': 'optimal_trade_entry_strategy',
        'SMT Divergence': 'smt_divergence_strategy',
        'Liquidity Voids / Inefficiencies': 'liquidity_voids_strategy'
      };
      
      const strategyKey = strategyMap[conceptName] || conceptName.toLowerCase()
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