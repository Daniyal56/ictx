import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  useTheme,
} from '@mui/material';
import { Link, useLocation } from 'react-router-dom';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

const Navbar: React.FC = () => {
  const theme = useTheme();
  const location = useLocation();

  const navItems = [
    { label: 'Dashboard', path: '/' },
    { label: 'Trading', path: '/trading' },
    { label: 'Backtesting', path: '/backtesting' },
    { label: 'Strategies', path: '/strategies' },
    { label: 'AI Insights', path: '/ai-insights' },
  ];

  return (
    <AppBar position="static" sx={{ backgroundColor: 'background.paper' }}>
      <Toolbar>
        <Box sx={{ display: 'flex', alignItems: 'center', mr: 4 }}>
          <TrendingUpIcon sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" component="div" sx={{ fontWeight: 'bold' }}>
            ICT Trading AI
          </Typography>
        </Box>
        
        <Box sx={{ flexGrow: 1, display: 'flex', gap: 2 }}>
          {navItems.map((item) => (
            <Button
              key={item.path}
              component={Link}
              to={item.path}
              sx={{
                color: location.pathname === item.path 
                  ? 'primary.main' 
                  : 'text.primary',
                backgroundColor: location.pathname === item.path 
                  ? 'rgba(0, 255, 136, 0.1)' 
                  : 'transparent',
                '&:hover': {
                  backgroundColor: 'rgba(0, 255, 136, 0.1)',
                },
              }}
            >
              {item.label}
            </Button>
          ))}
        </Box>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;