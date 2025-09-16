import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box } from '@mui/material';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import TradingView from './pages/TradingView';
import Backtesting from './pages/Backtesting';
import Strategies from './pages/Strategies';
import AIInsights from './pages/AIInsights';

function App() {
  return (
    <Box sx={{ 
      minHeight: '100vh', 
      backgroundColor: 'background.default',
      display: 'flex',
      flexDirection: 'column'
    }}>
      <Navbar />
      <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/trading" element={<TradingView />} />
          <Route path="/backtesting" element={<Backtesting />} />
          <Route path="/strategies" element={<Strategies />} />
          <Route path="/ai-insights" element={<AIInsights />} />
        </Routes>
      </Box>
    </Box>
  );
}

export default App;