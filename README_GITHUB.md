# Bond Sector Rotation Strategy Dashboard

A comprehensive Streamlit web application for analyzing bond sector rotation strategies using statistical mean reversion principles.

## Features

- **Interactive Strategy Selection**: Choose from 4 different bond sector rotation pairs
- **Real-time Data**: Fetches live market data using Yahoo Finance
- **Technical Analysis**: 4-panel charts showing price evolution, ratios, Z-scores, and returns
- **Performance Metrics**: Complete strategy performance analysis including Sharpe ratio, drawdowns, and win rates
- **Current Signals**: Live trading signals with clear buy/sell/neutral recommendations
- **Parameter Optimization**: Adjustable lookback windows and signal thresholds

## Available Strategies

1. **HYG/TLT**: High Yield vs Long Treasuries
2. **LQD/IEF**: Investment Grade vs Intermediate Treasuries  
3. **JNK/TLT**: Junk Bonds vs Long Treasuries
4. **AGG/TLT**: Aggregate Bonds vs Long Treasuries

## Live Demo

🌐 **[View Live App](https://bond-sector-rotation.streamlit.app)**

## Local Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bond-sector-rotation-strategy.git
cd bond-sector-rotation-strategy
```

2. Install dependencies:
```bash
pip install -r requirements_streamlit.txt
```

3. Run the application:
```bash
streamlit run bond_streamlit_app.py
```

4. Open your browser to `http://localhost:8501`

## How It Works

The application implements statistical mean reversion strategies:

1. **Calculate Price Ratio**: Asset1 / Asset2
2. **Compute Z-Score**: (Current Ratio - Rolling Mean) / Rolling Standard Deviation
3. **Generate Signals**: 
   - Long ratio when Z-Score < -Entry Threshold
   - Short ratio when Z-Score > +Entry Threshold
   - Exit when |Z-Score| < Exit Threshold

## Strategy Parameters

- **Lookback Window**: 20-252 days (controls sensitivity)
- **Entry Threshold**: 1.0-3.0 standard deviations (signal strength)
- **Exit Threshold**: 0.1-1.0 standard deviations (position holding period)

## Performance Metrics

- Total Return & Annualized Return
- Sharpe Ratio & Sortino Ratio
- Maximum Drawdown
- Win Rate & Total Trades
- Volatility Analysis

## File Structure

```
bond-sector-rotation-strategy/
├── bond_streamlit_app.py          # Main Streamlit application
├── launch_streamlit.py            # Application launcher
├── requirements_streamlit.txt     # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore file
```

## Technologies Used

- **Streamlit**: Web application framework
- **Plotly**: Interactive charts and visualizations
- **Pandas/NumPy**: Data analysis and computation
- **yfinance**: Financial data retrieval
- **Python 3.8+**: Programming language

## Disclaimer

This application is for educational and research purposes only. Past performance does not guarantee future results. Please consult with qualified financial professionals before implementing any investment strategies.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues, please open an issue on GitHub or contact the development team.