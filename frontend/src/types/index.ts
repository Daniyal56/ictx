export interface MarketData {
  symbol: string;
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'BUY' | 'SELL';
  price: number;
  quantity: number;
  timestamp: string;
  strategy: string;
  profit?: number;
  entry?: number;
  exit?: number;
  pnl?: number;
}

export interface BacktestRequest {
  strategy: string;
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  initial_capital?: number;
}

export interface BacktestResult {
  id: string;
  strategy: string;
  symbol: string;
  timeframe: string;
  start_date: string;
  end_date: string;
  initial_capital: number;
  final_capital: number;
  total_return: number;
  win_rate: number;
  max_drawdown: number;
  sharpe_ratio: number;
  total_trades: number;
  trades: Trade[];
  status: 'completed' | 'running' | 'failed';
}

export interface Strategy {
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
  complexity: 'Beginner' | 'Intermediate' | 'Advanced' | 'Expert';
  timeframe: string[];
  concepts: string[];
}

export interface MarketSentiment {
  symbol: string;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  confidence: number;
  signals: string[];
  aiScore: number;
}

export interface AIRecommendation {
  id: string;
  type: 'entry' | 'exit' | 'warning' | 'info';
  title: string;
  description: string;
  confidence: number;
  timestamp: string;
  symbol: string;
  priority: 'high' | 'medium' | 'low';
}

export interface PatternRecognition {
  pattern: string;
  probability: number;
  timeframe: string;
  symbol: string;
  description: string;
}

export interface DashboardStats {
  totalProfit: number;
  winRate: number;
  totalTrades: number;
  activeStrategies: number;
  portfolioValue: number;
  dailyPnL: number;
}

export interface OrderBlock {
  id: string;
  price: number;
  type: 'bullish' | 'bearish';
  strength: number;
  timestamp: string;
}

export interface FVG {
  id: string;
  high: number;
  low: number;
  type: 'bullish' | 'bearish';
  timestamp: string;
}

export interface ICTConcept {
  id: string;
  name: string;
  category: string;
  description: string;
  implementation: 'Basic' | 'Advanced' | 'Expert';
  isImplemented: boolean;
}

export interface ApiResponse<T> {
  data: T;
  message?: string;
  status: number;
}