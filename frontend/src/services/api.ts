const API_BASE_URL = 'http://localhost:8000';

export interface ApiResponse<T> {
  data: T;
  message?: string;
  status: number;
}

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
}

class ApiService {
  private async fetchApi<T>(endpoint: string, options?: RequestInit): Promise<T> {
    try {
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        headers: {
          'Content-Type': 'application/json',
          ...options?.headers,
        },
        ...options,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error(`API call failed: ${endpoint}`, error);
      throw error;
    }
  }

  // Market Data APIs
  async getMarketData(symbol: string, timeframe: string = '1h'): Promise<MarketData[]> {
    return this.fetchApi<MarketData[]>(`/api/market-data/${symbol}?timeframe=${timeframe}`);
  }

  async getMultipleMarketData(symbols: string[]): Promise<Record<string, MarketData[]>> {
    return this.fetchApi<Record<string, MarketData[]>>('/api/market-data/multiple', {
      method: 'POST',
      body: JSON.stringify({ symbols }),
    });
  }

  // Trading APIs
  async placeTrade(trade: Omit<Trade, 'id' | 'timestamp'>): Promise<Trade> {
    return this.fetchApi<Trade>('/api/trading/place-trade', {
      method: 'POST',
      body: JSON.stringify(trade),
    });
  }

  async getActiveTrades(): Promise<Trade[]> {
    return this.fetchApi<Trade[]>('/api/trading/active');
  }

  async getTradeHistory(): Promise<Trade[]> {
    return this.fetchApi<Trade[]>('/api/trading/history');
  }

  // Backtesting APIs
  async runBacktest(request: BacktestRequest): Promise<BacktestResult> {
    return this.fetchApi<BacktestResult>('/api/backtesting/run', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async getBacktestResults(): Promise<BacktestResult[]> {
    return this.fetchApi<BacktestResult[]>('/api/backtesting/results');
  }

  async getBacktestResult(id: string): Promise<BacktestResult> {
    return this.fetchApi<BacktestResult>(`/api/backtesting/results/${id}`);
  }

  // AI Agent APIs
  async getMarketAnalysis(symbols: string[]): Promise<any> {
    return this.fetchApi('/api/ai-agent/market-analysis', {
      method: 'POST',
      body: JSON.stringify({ symbols }),
    });
  }

  async getAIRecommendations(): Promise<any> {
    return this.fetchApi('/api/ai-agent/recommendations');
  }

  async getPatternRecognition(symbol: string, timeframe: string): Promise<any> {
    return this.fetchApi(`/api/ai-agent/pattern-recognition?symbol=${symbol}&timeframe=${timeframe}`);
  }

  async getSentimentAnalysis(): Promise<any> {
    return this.fetchApi('/api/ai-agent/sentiment');
  }

  // Strategy APIs
  async getAvailableStrategies(): Promise<any> {
    return this.fetchApi('/api/strategies/available');
  }

  async getStrategyPerformance(strategy: string): Promise<any> {
    return this.fetchApi(`/api/strategies/performance/${strategy}`);
  }

  async updateStrategyParameters(strategy: string, parameters: any): Promise<any> {
    return this.fetchApi(`/api/strategies/update/${strategy}`, {
      method: 'PUT',
      body: JSON.stringify(parameters),
    });
  }

  // Health check
  async healthCheck(): Promise<{ status: string; message: string }> {
    return this.fetchApi('/health');
  }
}

export const apiService = new ApiService();
export default apiService;