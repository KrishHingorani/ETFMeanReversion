# Portfolio Optimization Tool - Streamlit App

A Modern Portfolio Theory (MPT) based portfolio optimization tool built with Streamlit. This app helps optimize portfolio allocations for ETFs and stocks using various optimization objectives.

## Features

- **Multiple Optimization Objectives**: Sharpe ratio maximization, minimum variance, drawdown minimization, equal weighting
- **Comprehensive Metrics**: Annual return, volatility, Sharpe ratio, Calmar ratio, Sortino ratio, VaR, CVaR
- **Interactive Visualizations**: Portfolio allocation charts, cumulative returns, drawdown analysis
- **Real-time Data**: Fetches live market data from Yahoo Finance
- **Auto-Fetching Portfolio Factsheet**: Comprehensive symbol information (market cap, P/E, beta, sector, etc.)
- **Auto-Fetching Options IV Analysis**: Automatic implied volatility analysis for individual stocks using AlphaQuery data
- **Export Capabilities**: Download results as Excel or CSV files
- **Clean UI**: White and blue color scheme inspired by PrairieLearn

## Installation & Local Development

1. Clone this repository or download the files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app locally:
   ```bash
   streamlit run app.py
   ```

## Deployment on Streamlit Cloud

1. Push this folder to your GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository and this folder
5. Set the main file path as `app.py`
6. Deploy!

## Usage

1. **Enter Symbols**: Input ETF/stock symbols (comma-separated) in the sidebar
2. **Choose Objective**: Select your optimization strategy
3. **Set Parameters**: Choose data period
4. **Optimize**: Click "Optimize Portfolio" to run the analysis
5. **Review Results**: Examine allocation, metrics, and charts
6. **Analyze Options**: Use the IV analysis section to examine implied volatility for stocks
7. **Export**: Download results in Excel or CSV format

## Supported Symbols

Any valid Yahoo Finance ticker symbols including:
- ETFs: SPY, QQQ, VTI, IWM, etc.
- Stocks: AAPL, GOOGL, MSFT, TSLA, etc.
- International: Use appropriate Yahoo Finance symbols

## Optimization Methods

- **Equal Weighted**: Simple 1/n allocation
- **Sharpe Ratio**: Maximize risk-adjusted returns
- **Minimum Variance**: Minimize portfolio volatility
- **Drawdown**: Minimize maximum drawdown
- **Risk-Adjusted Drawdown**: Optimize drawdown relative to volatility

## Technical Details

- Built with Streamlit for the web interface
- Uses yfinance for market data
- Scipy for optimization algorithms
- Plotly for interactive charts
- Modern Portfolio Theory for calculations

## Notes

- Historical performance does not guarantee future results
- This tool is for educational and informational purposes
- Always consult with financial professionals for investment decisions
- Market data is subject to delays and may not be real-time

## Author

Created by Krish Hingorani