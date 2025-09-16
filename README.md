# ICT Trading AI Agent

A comprehensive full-stack ICT (Inner Circle Trader) trading AI agent application featuring advanced trading strategies, backtesting engine, and AI-powered decision making.

## Features

- 🚀 Python FastAPI backend with advanced trading strategies
- ⚛️ React TypeScript frontend with modern dashboard
- 📊 Comprehensive backtesting engine
- 🧠 50+ ICT trading concepts implementation
- 🤖 AI-powered decision making and pattern recognition
- 📈 Real-time market analysis and visualization

## Project Structure

```
ictx/
├── backend/          # FastAPI backend
│   ├── app/          # Main application
│   ├── strategies/   # ICT trading strategies
│   ├── backtesting/  # Backtesting engine
│   └── ai/          # AI agent logic
├── frontend/         # React TypeScript frontend
├── data/            # Market data and datasets
├── docs/            # Documentation
└── tests/           # Test suites
```

## ICT Concepts Implemented

### Core ICT Concepts
- Market Structure (HH, HL, LH, LL)
- Liquidity (buy-side & sell-side)
- Liquidity Pools
- Order Blocks (Bullish & Bearish)
- Breaker Blocks
- Fair Value Gaps (FVG) / Imbalances
- Rejection Blocks
- Mitigation Blocks
- Supply & Demand Zones
- Premium & Discount (OTE)

### Time & Price Theory
- Killzones (London, New York, Asia)
- Session Opens and Midnight Open
- Equilibrium & Fibonacci Ratios
- Daily & Weekly Range Expectations
- Session Liquidity Raids

### Advanced Strategies
- ICT Silver Bullet
- Asian Range Breakout
- New York Reversal
- London Killzone Strategy
- FVG Sniper Entry
- Order Block Strategy
- SMT Divergence Strategy
- Power of 3 Model

## Getting Started

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm start
```

## License

MIT License