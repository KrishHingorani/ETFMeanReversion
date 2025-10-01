# Streamlit Deployment Guide

## Quick Start

1. **Test Locally First**:
   ```bash
   cd streamlit_portfolio_app
   pip install -r requirements.txt
   streamlit run app.py
   ```

2. **Deploy to Streamlit Cloud**:
   - Push this folder to your GitHub repository
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect GitHub and select your repo
   - Set main file as `app.py`
   - Deploy!

## File Structure
```
streamlit_portfolio_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # Documentation
├── test_app.py        # Test script
└── deploy_guide.md    # This file
```

## Key Differences from Original Dash App

✅ **Removed**:
- Bloomberg yellow color scheme and terminal styling
- Complex CSS customizations
- Dash-specific components
- AlphaQuery IV scraper dependency (made optional)

✅ **Added**:
- Clean white/blue PrairieLearn-inspired design
- Streamlit-native components
- Better mobile responsiveness
- Simplified deployment process

✅ **Maintained**:
- All portfolio optimization logic
- Modern Portfolio Theory calculations
- Multiple optimization objectives
- Performance metrics calculations
- Export functionality
- Real-time data fetching

## Customization Options

1. **Colors**: Modify the CSS in `app.py` to change the color scheme
2. **Metrics**: Add more performance metrics in the `calculate_portfolio_metrics` function
3. **Charts**: Customize Plotly charts in the visualization functions
4. **Export**: Add more export formats (PDF, etc.)

## Troubleshooting

- **Import Errors**: Ensure all packages in requirements.txt are installed
- **Data Issues**: Check Yahoo Finance symbol validity
- **Memory Issues**: Reduce data period or number of symbols for large portfolios
- **Performance**: Consider caching with `@st.cache_data` for production use

## Production Considerations

1. Add error handling for network issues
2. Implement caching for better performance
3. Add user authentication if needed
4. Consider rate limiting for API calls
5. Add logging for debugging

Enjoy your new Streamlit portfolio optimization tool! 🚀