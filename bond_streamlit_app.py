"""
Bond Sector Rotation Strategy - Streamlit Dashboard
==================================================

Interactive Streamlit application for bond sector rotation strategies:
- HYG/TLT: High Yield vs Long Treasuries
- LQD/IEF: Investment Grade vs Intermediate Treasuries  
- JNK/TLT: Junk Bonds vs Long Treasuries
- AGG/TLT: Aggregate Bonds vs Long Treasuries

Author: Research Team
Date: September 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Bond Sector Rotation Strategy",
    page_icon="BS",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bond strategy configurations
BOND_STRATEGIES = {
    'HYG_TLT': {
        'symbols': ['HYG', 'TLT'],
        'name': 'High Yield vs Long Treasuries',
        'description': 'Credit risk strategy: High yield corporate bonds vs long-term government bonds',
        'color': '#FF6B6B'
    },
    'LQD_IEF': {
        'symbols': ['LQD', 'IEF'],
        'name': 'Investment Grade vs Intermediate Treasuries',
        'description': 'Credit quality strategy: Investment grade corporates vs intermediate treasuries',
        'color': '#4ECDC4'
    },
    'JNK_TLT': {
        'symbols': ['JNK', 'TLT'],
        'name': 'Junk Bonds vs Long Treasuries',
        'description': 'High risk credit strategy: Junk bonds vs long-term treasuries',
        'color': '#45B7D1'
    },
    'AGG_TLT': {
        'symbols': ['AGG', 'TLT'],
        'name': 'Aggregate Bonds vs Long Treasuries',
        'description': 'Broad bond strategy: Total bond market vs long-term treasuries',
        'color': '#96CEB4'
    }
}

class BondStrategyEngine:
    """Bond strategy calculation engine"""
    
    def __init__(self, lookback_years=5):
        self.lookback_years = lookback_years
        
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def fetch_data(_self, symbols):
        """Fetch market data with caching"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=_self.lookback_years * 365)
            
            with st.spinner(f"Fetching data for {', '.join(symbols)}..."):
                data = yf.download(symbols, start=start_date, end=end_date, progress=False)
            
            if isinstance(data.columns, pd.MultiIndex):
                data = data['Close']
            elif isinstance(data, pd.Series):
                data = data.to_frame(symbols[0])
            
            data = data.fillna(method='ffill').fillna(method='bfill')
            
            return data
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_strategy(self, asset1_symbol, asset2_symbol, window=60, 
                          entry_threshold=2.0, exit_threshold=0.5):
        """Calculate complete strategy analysis"""
        try:
            symbols = [asset1_symbol, asset2_symbol]
            data = self.fetch_data(symbols)
            
            if data.empty or asset1_symbol not in data.columns or asset2_symbol not in data.columns:
                return {'error': f'Data not available for {asset1_symbol}/{asset2_symbol}'}
            
            price1 = data[asset1_symbol].dropna()
            price2 = data[asset2_symbol].dropna()
            
            common_dates = price1.index.intersection(price2.index)
            if len(common_dates) < window:
                return {'error': f'Insufficient data points: {len(common_dates)}'}
            
            price1_aligned = price1[common_dates]
            price2_aligned = price2[common_dates]
            
            # Calculate ratio and z-score
            ratio = price1_aligned / price2_aligned
            rolling_mean = ratio.rolling(window=window).mean()
            rolling_std = ratio.rolling(window=window).std()
            zscore = (ratio - rolling_mean) / rolling_std
            
            # Generate signals
            signals = pd.Series(0, index=zscore.index)
            position = 0
            
            for i in range(1, len(zscore)):
                if pd.isna(zscore.iloc[i]):
                    signals.iloc[i] = position
                    continue
                    
                current_zscore = zscore.iloc[i]
                
                if position == 0:
                    if current_zscore < -entry_threshold:
                        position = 1  # Long ratio
                    elif current_zscore > entry_threshold:
                        position = -1  # Short ratio
                elif abs(current_zscore) < exit_threshold:
                    position = 0
                
                signals.iloc[i] = position
            
            # Calculate returns
            returns1 = price1_aligned.pct_change()
            returns2 = price2_aligned.pct_change()
            strategy_returns = signals.shift(1) * (returns1 - returns2)
            strategy_returns = strategy_returns.fillna(0)
            
            # Performance metrics
            total_return = (1 + strategy_returns).prod() - 1
            annualized_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            annualized_vol = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
            
            cumulative = (1 + strategy_returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            winning_trades = (strategy_returns > 0).sum()
            total_trades = (strategy_returns != 0).sum()
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            downside_returns = strategy_returns[strategy_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = annualized_return / downside_deviation if downside_deviation > 0 else 0
            
            return {
                'success': True,
                'ratio': ratio,
                'zscore': zscore,
                'signals': signals,
                'returns': strategy_returns,
                'cumulative_returns': cumulative,
                'price1': price1_aligned,
                'price2': price2_aligned,
                'metrics': {
                    'Total Return': total_return,
                    'Annualized Return': annualized_return,
                    'Annualized Volatility': annualized_vol,
                    'Sharpe Ratio': sharpe_ratio,
                    'Sortino Ratio': sortino_ratio,
                    'Max Drawdown': max_drawdown,
                    'Win Rate': win_rate,
                    'Total Trades': total_trades
                },
                'current_signals': {
                    'ratio': ratio.iloc[-1] if len(ratio) > 0 else np.nan,
                    'zscore': zscore.iloc[-1] if len(zscore) > 0 else np.nan,
                    'signal': signals.iloc[-1] if len(signals) > 0 else 0
                },
                'parameters': {
                    'window': window,
                    'entry_threshold': entry_threshold,
                    'exit_threshold': exit_threshold
                }
            }
            
        except Exception as e:
            return {'error': f'Strategy calculation failed: {str(e)}'}

# Initialize engine
if 'engine' not in st.session_state:
    st.session_state.engine = BondStrategyEngine()

def main():
    """Main application function"""
    
    # Header
    st.title("Bond Sector Rotation Strategy Dashboard")
    st.markdown("### Analyze credit cycle and duration risk strategies across bond sectors")
    
    # Sidebar - Strategy Selection
    st.sidebar.header("Strategy Configuration")
    
    # Strategy selection
    strategy_options = list(BOND_STRATEGIES.keys())
    strategy_labels = [f"{BOND_STRATEGIES[k]['name']} ({'/'.join(BOND_STRATEGIES[k]['symbols'])})" 
                      for k in strategy_options]
    
    selected_strategy = st.sidebar.selectbox(
        "Select Bond Strategy:",
        options=strategy_options,
        format_func=lambda x: f"{BOND_STRATEGIES[x]['name']} ({'/'.join(BOND_STRATEGIES[x]['symbols'])})",
        index=0
    )
    
    # Parameter controls
    st.sidebar.subheader("Parameters")
    
    window = st.sidebar.slider(
        "Lookback Window (days):",
        min_value=20, max_value=252, value=60, step=10,
        help="Rolling window for calculating statistics"
    )
    
    entry_threshold = st.sidebar.slider(
        "Entry Threshold (σ):",
        min_value=1.0, max_value=3.0, value=2.0, step=0.25,
        help="Z-score threshold for entering positions"
    )
    
    exit_threshold = st.sidebar.slider(
        "Exit Threshold (σ):",
        min_value=0.1, max_value=1.0, value=0.5, step=0.1,
        help="Z-score threshold for exiting positions"
    )
    
    # Refresh button
    if st.sidebar.button("Refresh Data", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Get strategy configuration
    config = BOND_STRATEGIES[selected_strategy]
    symbols = config['symbols']
    
    # Calculate strategy
    with st.spinner("Calculating strategy..."):
        result = st.session_state.engine.calculate_strategy(
            symbols[0], symbols[1], 
            window, entry_threshold, exit_threshold
        )
    
    if result.get('error'):
        st.error(f"Error: {result['error']}")
        return
    
    # Display strategy info
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"{config['name']}")
        st.write(config['description'])
        st.write(f"**Symbols:** {' vs '.join(symbols)}")
    
    with col2:
        # Current signal badge
        current_signals = result.get('current_signals', {})
        signal_val = current_signals.get('signal', 0)
        zscore_val = current_signals.get('zscore', 0)
        
        if signal_val > 0:
            st.success(f"LONG {symbols[0]}")
            st.write(f"**Action:** Buy {symbols[0]}, Sell {symbols[1]}")
        elif signal_val < 0:
            st.error(f"SHORT {symbols[0]}")
            st.write(f"**Action:** Sell {symbols[0]}, Buy {symbols[1]}")
        else:
            st.warning("NEUTRAL")
            st.write("**Action:** No position")
        
        st.write(f"**Current Z-Score:** {zscore_val:+.2f}")
    
    # Performance metrics
    metrics = result.get('metrics', {})
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Return",
            f"{metrics.get('Total Return', 0):.1%}",
            help="Total strategy return over the period"
        )
    
    with col2:
        st.metric(
            "Sharpe Ratio",
            f"{metrics.get('Sharpe Ratio', 0):.2f}",
            help="Risk-adjusted return measure"
        )
    
    with col3:
        st.metric(
            "Max Drawdown",
            f"{metrics.get('Max Drawdown', 0):.1%}",
            help="Maximum peak-to-trough decline"
        )
    
    with col4:
        st.metric(
            "Win Rate",
            f"{metrics.get('Win Rate', 0):.1%}",
            help="Percentage of profitable trades"
        )
    
    # Charts
    st.subheader("Strategy Analysis")
    
    # Create 4-panel chart
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=[
            f'Price Evolution: {" vs ".join(symbols)}',
            'Price Ratio',
            'Z-Score with Entry/Exit Thresholds',
            'Cumulative Strategy Returns'
        ],
        vertical_spacing=0.08
    )
    
    # Get data
    ratio = result['ratio']
    zscore = result['zscore']
    signals = result['signals']
    cumulative = result['cumulative_returns']
    price1 = result['price1']
    price2 = result['price2']
    
    # 1. Price Evolution
    fig.add_trace(
        go.Scatter(x=price1.index, y=price1.values, 
                  name=symbols[0], line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=price2.index, y=price2.values, 
                  name=symbols[1], line=dict(color='red')),
        row=1, col=1
    )
    
    # 2. Price Ratio
    fig.add_trace(
        go.Scatter(x=ratio.index, y=ratio.values, 
                  name='Price Ratio', line=dict(color='purple')),
        row=2, col=1
    )
    
    # 3. Z-Score with thresholds
    fig.add_trace(
        go.Scatter(x=zscore.index, y=zscore.values, 
                  name='Z-Score', line=dict(color='black')),
        row=3, col=1
    )
    
    fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=exit_threshold, line_dash="dot", line_color="orange", row=3, col=1)
    fig.add_hline(y=-exit_threshold, line_dash="dot", line_color="orange", row=3, col=1)
    fig.add_hline(y=0, line_dash="solid", line_color="gray", row=3, col=1)
    
    # 4. Cumulative Returns
    fig.add_trace(
        go.Scatter(x=cumulative.index, y=cumulative.values, 
                  name='Strategy Returns', line=dict(color='green', width=2)),
        row=4, col=1
    )
    
    fig.update_layout(height=800, showlegend=True, title_text=f"Bond Strategy Analysis: {config['name']}")
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance Table
    st.subheader("Detailed Performance Metrics")
    
    performance_data = {
        "Metric": [
            "Total Return", "Annualized Return", "Annualized Volatility",
            "Sharpe Ratio", "Sortino Ratio", "Max Drawdown", 
            "Win Rate", "Total Trades"
        ],
        "Value": [
            f"{metrics.get('Total Return', 0):.2%}",
            f"{metrics.get('Annualized Return', 0):.2%}",
            f"{metrics.get('Annualized Volatility', 0):.2%}",
            f"{metrics.get('Sharpe Ratio', 0):.3f}",
            f"{metrics.get('Sortino Ratio', 0):.3f}",
            f"{metrics.get('Max Drawdown', 0):.2%}",
            f"{metrics.get('Win Rate', 0):.1%}",
            f"{metrics.get('Total Trades', 0):.0f}"
        ]
    }
    
    performance_df = pd.DataFrame(performance_data)
    st.dataframe(performance_df, use_container_width=True, hide_index=True)
    
    # Strategy Parameters
    with st.expander("Current Strategy Parameters"):
        params = result.get('parameters', {})
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Lookback Window:** {params.get('window', 0)} days")
        with col2:
            st.write(f"**Entry Threshold:** ±{params.get('entry_threshold', 0)} σ")
        with col3:
            st.write(f"**Exit Threshold:** ±{params.get('exit_threshold', 0)} σ")
    
    # Footer
    st.markdown("---")
    st.markdown("**Disclaimer:** This application is for educational and research purposes only. Past performance does not guarantee future results.")
    
    # Auto-refresh
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.caption(f"Last updated: {timestamp}")

if __name__ == "__main__":
    main()