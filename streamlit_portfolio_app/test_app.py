#!/usr/bin/env python3
"""
Quick test script to verify the core functionality of the portfolio optimizer
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from app import ETFPortfolioOptimizer
    import yfinance as yf
    import pandas as pd
    import numpy as np
    
    print("✅ All imports successful")
    
    # Test basic functionality
    test_symbols = ['SPY', 'QQQ']
    print(f"🧪 Testing with symbols: {test_symbols}")
    
    optimizer = ETFPortfolioOptimizer(test_symbols)
    print("✅ Optimizer initialized")
    
    # Test data fetch (small period for speed)
    data = optimizer.fetch_data(period='1mo')
    print(f"✅ Data fetched: {data.shape[0]} rows, {data.shape[1]} columns")
    
    # Test returns calculation
    returns = optimizer.calculate_returns()
    print(f"✅ Returns calculated: {returns.shape[0]} rows")
    
    # Test optimization
    weights = optimizer.optimize_portfolio(objective='equal_weighted')
    print(f"✅ Optimization successful: {weights}")
    print(f"   Weights sum to: {weights.sum():.4f}")
    
    # Test metrics calculation
    metrics = optimizer.calculate_portfolio_metrics(weights)
    print(f"✅ Metrics calculated:")
    print(f"   Annual Return: {metrics['annual_return']*100:.2f}%")
    print(f"   Annual Volatility: {metrics['annual_volatility']*100:.2f}%")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    
    print("\n🎉 All tests passed! The app should work correctly.")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)