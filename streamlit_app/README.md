# Bond Sector Rotation Strategy - Streamlit App

A comprehensive Streamlit web application for analyzing bond sector rotation strategies using statistical mean reversion principles.

## Quick Start

```bash
# From this directory
pip install -r requirements.txt
streamlit run bond_streamlit_app.py
```

## Files in this folder:

- `bond_streamlit_app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies for deployment
- `launch_streamlit.py` - Application launcher script
- `README_GITHUB.md` - Detailed GitHub documentation
- `.gitignore` - Git ignore configuration

## Available Bond Strategies

1. **HYG/TLT**: High Yield vs Long Treasuries
2. **LQD/IEF**: Investment Grade vs Intermediate Treasuries  
3. **JNK/TLT**: Junk Bonds vs Long Treasuries
4. **AGG/TLT**: Aggregate Bonds vs Long Treasuries

## Features

- Interactive strategy selection and parameter controls
- Real-time market data fetching with yfinance
- 4-panel technical analysis charts
- Performance metrics and current trading signals
- Clean, professional interface without emojis

## Deployment

Ready for deployment on:
- Streamlit Cloud (recommended)
- Heroku
- Railway  
- Any Python hosting platform

## Usage

1. Select a bond strategy from the dropdown
2. Adjust parameters (lookback window, thresholds)
3. View real-time analysis and signals
4. Monitor performance metrics