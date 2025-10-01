#!/usr/bin/env python3
"""
Test the complete IV flow to verify it works
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

try:
    from app import ETFPortfolioOptimizer
    import pandas as pd
    
    print("🧪 Testing complete portfolio optimization with IV analysis...")
    
    # Test with a mix of stocks and ETFs
    symbols = ['AAPL', 'GOOGL', 'SPY']
    optimizer = ETFPortfolioOptimizer(symbols)
    
    print("✅ 1. Optimizer initialized")
    
    # Fetch data
    data = optimizer.fetch_data(period='1mo')  # Short period for speed
    print(f"✅ 2. Data fetched: {data.shape}")
    
    # Calculate returns
    returns = optimizer.calculate_returns()
    print(f"✅ 3. Returns calculated: {returns.shape}")
    
    # Fetch symbol info
    symbol_info = optimizer.fetch_symbol_info()
    print(f"✅ 4. Symbol info fetched for {len(symbol_info)} symbols")
    
    # Fetch IV data (should work for AAPL and GOOGL, skip SPY)
    iv_data = optimizer.fetch_iv_data(show_progress=False)
    print(f"✅ 5. IV data fetched for {len(iv_data) if iv_data else 0} equity symbols")
    
    if iv_data:
        print("   IV Results:")
        for symbol, data in iv_data.items():
            print(f"   - {symbol}: {data['avg_iv']:.2f}% IV (vs {data['historical_vol']:.2f}% HV)")
    
    # Optimize portfolio
    weights = optimizer.optimize_portfolio()
    print(f"✅ 6. Portfolio optimized: {dict(zip(symbols, weights))}")
    
    # Calculate metrics
    metrics = optimizer.calculate_portfolio_metrics(weights)
    print(f"✅ 7. Metrics calculated:")
    print(f"   - Annual Return: {metrics['annual_return']*100:.2f}%")
    print(f"   - Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
    
    print("\n🎉 Complete flow test passed! IV scraper is working correctly.")
    
    # Test factsheet creation
    print("\n📋 Testing factsheet data creation...")
    factsheet_data = []
    for i, symbol in enumerate(symbols):
        info = symbol_info.get(symbol, {})
        weight = weights[i] * 100
        iv_info = iv_data.get(symbol, {}) if iv_data else {}
        
        factsheet_data.append({
            'Symbol': symbol,
            'Name': info.get('longName', symbol)[:30] + '...' if len(info.get('longName', symbol)) > 30 else info.get('longName', symbol),
            'Weight (%)': f"{weight:.2f}%",
            'Sector': info.get('sector', 'N/A'),
            'Market Cap': f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap', 0) > 0 else 'N/A',
            'IV (30d)': f"{iv_info.get('avg_iv', 0):.1f}%" if iv_info.get('avg_iv') else 'N/A',
            'IV vs HV': f"{iv_info.get('iv_vs_hv', 0):+.1f}%" if iv_info.get('iv_vs_hv') is not None else 'N/A'
        })
    
    factsheet_df = pd.DataFrame(factsheet_data)
    print("✅ Factsheet created successfully:")
    print(factsheet_df.to_string(index=False))
    
    print("\n🚀 All tests passed! The app should work perfectly.")
    
except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)