import streamlit as st
import plotly.graph_objects as go
import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import IV scraper (optional)
try:
    from alphaquery_iv_scraper import AlphaQueryIVScraper
except ImportError:
    AlphaQueryIVScraper = None

# Export libraries
import io

# Configure Streamlit page
st.set_page_config(
    page_title="Portfolio Optimization Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple, clean styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-title {
        color: #007bff;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .section-title {
        color: #007bff;
        font-weight: bold;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #007bff;
        padding-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class ETFPortfolioOptimizer:
    def __init__(self, etf_symbols):
        self.etf_symbols = etf_symbols
        self.data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.benchmark_data = None
        self.benchmark_returns = None
        if AlphaQueryIVScraper:
            self.iv_scraper = AlphaQueryIVScraper()
        else:
            self.iv_scraper = None
        self.iv_data = None

    def fetch_data(self, period='2y'):
        """Fetch historical data for ETFs using yfinance including dividends"""
        try:
            # Download data with auto_adjust=True for dividend-adjusted prices
            # This automatically includes dividend reinvestment in the price series
            raw_data = yf.download(self.etf_symbols, period=period, progress=False, auto_adjust=True)

            # Handle single vs multiple symbols
            if len(self.etf_symbols) == 1:
                # For single symbol, use Close price (already dividend-adjusted)
                self.data = raw_data['Close'].to_frame(self.etf_symbols[0])
            else:
                # For multiple symbols, use Close prices (already dividend-adjusted)
                if isinstance(raw_data.columns, pd.MultiIndex):
                    self.data = raw_data['Close']
                else:
                    self.data = raw_data[['Close']].rename(columns={'Close': self.etf_symbols[0]})

            # Clean data
            self.data = self.data.dropna()

            # Validate we have data
            if self.data.empty:
                raise ValueError("No data available for the specified ETFs and period")

            return self.data

        except Exception as e:
            st.error(f"Error in main fetch, trying fallback method: {e}")
            # Fallback: Try individual downloads with dividend-adjusted prices
            valid_data = {}
            for symbol in self.etf_symbols:
                try:
                    # Try to get individual symbol data with auto-adjustment (includes dividends)
                    ticker = yf.Ticker(symbol)
                    hist_data = ticker.history(period=period, auto_adjust=True)

                    if not hist_data.empty and 'Close' in hist_data.columns:
                        valid_data[symbol] = hist_data['Close']

                        # Get dividend info for logging
                        div_data = ticker.history(period=period, auto_adjust=False)
                        if not div_data.empty and 'Dividends' in div_data.columns:
                            total_divs = div_data['Dividends'].sum()
                            st.info(f"{symbol}: Data loaded with dividend adjustment (Total dividends: ${total_divs:.4f})")
                        else:
                            st.info(f"{symbol}: Data loaded")
                    else:
                        st.warning(f"{symbol}: No valid data found")

                except Exception as symbol_error:
                    st.error(f"{symbol}: Failed to fetch data - {symbol_error}")
                    continue

            if not valid_data:
                raise ValueError("No valid data could be fetched for any of the specified symbols")

            # Combine valid data
            self.data = pd.DataFrame(valid_data)
            self.data = self.data.dropna()

            if self.data.empty:
                raise ValueError("No overlapping data available for the specified symbols")

            st.success(f"Successfully loaded data for {len(self.data.columns)} symbols")
            return self.data

    def calculate_returns(self):
        """Calculate daily returns"""
        if self.data is None:
            raise ValueError("No data available. Please fetch data first.")
        
        self.returns = self.data.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        return self.returns

    def fetch_benchmark_data(self, period='2y'):
        """Fetch benchmark data for comparison"""
        benchmark_symbols = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']
        try:
            benchmark_data = yf.download(benchmark_symbols, period=period, progress=False, auto_adjust=True)
            
            if isinstance(benchmark_data.columns, pd.MultiIndex):
                self.benchmark_data = benchmark_data['Close']
            else:
                # Handle single benchmark case
                self.benchmark_data = benchmark_data
            
            self.benchmark_data = self.benchmark_data.dropna()
            self.benchmark_returns = self.benchmark_data.pct_change().dropna()
            
            return self.benchmark_data
        except Exception as e:
            st.warning(f"Could not fetch benchmark data: {e}")
            return None

    def fetch_factor_correlation(self, period='2y'):
        """Fetch factor ETF data for correlation analysis"""
        factor_etfs = ['SPMO', 'RPV', 'QVML', 'QUAL', 'VFMF']
        try:
            factor_data = yf.download(factor_etfs, period=period, progress=False, auto_adjust=True)
            
            if isinstance(factor_data.columns, pd.MultiIndex):
                self.factor_data = factor_data['Close']
            else:
                self.factor_data = factor_data
            
            # Filter out ETFs with insufficient data (less than 50 days)
            valid_etfs = []
            for etf in self.factor_data.columns:
                etf_data = self.factor_data[etf].dropna()
                if len(etf_data) >= 50:  # Minimum data threshold
                    valid_etfs.append(etf)
            
            if valid_etfs:
                self.factor_data = self.factor_data[valid_etfs]
                self.factor_data = self.factor_data.dropna()
                self.factor_returns = self.factor_data.pct_change().dropna()
            else:
                self.factor_returns = None
            
            return self.factor_data
        except Exception as e:
            if hasattr(st, 'warning'):
                st.warning(f"Could not fetch factor data: {e}")
            return None

    def calculate_factor_correlations(self, weights):
        """Calculate correlations between portfolio and factor ETFs"""
        if not hasattr(self, 'factor_returns') or self.factor_returns is None:
            return None
        
        # Calculate portfolio returns
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        # Align dates
        common_dates = portfolio_returns.index.intersection(self.factor_returns.index)
        if len(common_dates) < 20:  # Need minimum data points
            return None
        
        portfolio_aligned = portfolio_returns.loc[common_dates]
        factor_aligned = self.factor_returns.loc[common_dates]
        
        # Calculate correlations
        correlations = {}
        factor_names = {
            'SPMO': 'Momentum',
            'RPV': 'Value', 
            'QVML': 'Multi Factor',
            'QUAL': 'Quality',
            'VFMF': 'Min Volatility'
        }
        
        for factor in factor_aligned.columns:
            corr = portfolio_aligned.corr(factor_aligned[factor])
            correlations[factor_names.get(factor, factor)] = corr
        
        return correlations

    def calculate_pca_factor_decomposition(self, n_components=None):
        """Perform PCA factor decomposition on portfolio returns"""
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            if self.returns is None:
                return None
            
            # Standardize the returns for PCA
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(self.returns)
            
            # Default to min(n_assets, n_periods/4) components
            if n_components is None:
                n_components = min(len(self.etf_symbols), len(self.returns) // 4, 10)
            
            # Perform PCA
            pca = PCA(n_components=n_components)
            pca_factors = pca.fit_transform(returns_scaled)
            
            # Create results dictionary
            pca_results = {
                'n_components': n_components,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance_explained': np.cumsum(pca.explained_variance_ratio_),
                'components': pca.components_,  # Factor loadings
                'factor_returns': pca_factors,
                'symbols': self.etf_symbols,
                'total_variance_explained': np.sum(pca.explained_variance_ratio_)
            }
            
            # Calculate factor contributions for each asset
            factor_contributions = {}
            for i, symbol in enumerate(self.etf_symbols):
                contributions = []
                for j in range(n_components):
                    # Factor loading * explained variance
                    contribution = abs(pca.components_[j, i]) * pca.explained_variance_ratio_[j]
                    contributions.append(contribution)
                factor_contributions[symbol] = contributions
            
            pca_results['factor_contributions'] = factor_contributions
            
            # Add financial interpretation for each factor
            pca_results['factor_interpretations'] = self._interpret_pca_factors(pca_results)
            
            return pca_results
            
        except ImportError:
            if hasattr(st, 'error'):
                st.error("scikit-learn is required for PCA analysis. Please install it: pip install scikit-learn")
            return None
        except Exception as e:
            if hasattr(st, 'error'):
                st.error(f"Error in PCA factor decomposition: {str(e)}")
            return None

    def _interpret_pca_factors(self, pca_results):
        """Interpret PCA factors in financial terms"""
        try:
            import yfinance as yf
            
            interpretations = {}
            symbols = pca_results['symbols']
            components = pca_results['components']
            
            # Get asset characteristics for interpretation
            asset_info = {}
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    asset_info[symbol] = {
                        'sector': info.get('sector', 'Unknown'),
                        'quote_type': info.get('quoteType', 'Unknown'),
                        'market_cap': info.get('marketCap', 0),
                        'name': info.get('longName', symbol)
                    }
                except:
                    asset_info[symbol] = {
                        'sector': 'Unknown',
                        'quote_type': 'Unknown', 
                        'market_cap': 0,
                        'name': symbol
                    }
            
            # Interpret each component
            for i in range(pca_results['n_components']):
                loadings = components[i]
                variance_explained = pca_results['explained_variance_ratio'][i]
                
                # Find dominant assets (highest absolute loadings)
                loading_pairs = [(abs(loadings[j]), symbols[j], loadings[j]) for j in range(len(symbols))]
                loading_pairs.sort(reverse=True)
                
                # Get top contributors
                top_contributors = loading_pairs[:3]
                
                # Determine factor type based on loadings and asset characteristics
                interpretation = self._determine_factor_type(i, loadings, symbols, asset_info, variance_explained)
                
                interpretations[f'PC{i+1}'] = {
                    'financial_meaning': interpretation['meaning'],
                    'explanation': interpretation['explanation'],
                    'top_drivers': [{'symbol': symbol, 'loading': loading, 'name': asset_info[symbol]['name']} 
                                   for _, symbol, loading in top_contributors],
                    'variance_explained': variance_explained,
                    'factor_type': interpretation['factor_type']
                }
            
            return interpretations
            
        except Exception:
            return {}

    def _determine_factor_type(self, component_idx, loadings, symbols, asset_info, variance_explained):
        """Determine what financial factor this component represents"""
        
        # Analyze loading patterns
        all_positive = all(loading > 0.1 for loading in loadings)
        all_negative = all(loading < -0.1 for loading in loadings)
        mixed_signs = not all_positive and not all_negative
        
        # Get asset types
        etf_count = sum(1 for symbol in symbols if asset_info[symbol]['quote_type'] == 'ETF')
        stock_count = len(symbols) - etf_count
        
        # Get sectors represented
        sectors = [asset_info[symbol]['sector'] for symbol in symbols if asset_info[symbol]['sector'] != 'Unknown']
        unique_sectors = len(set(sectors))
        
        # Determine factor type based on component index and characteristics
        if component_idx == 0:  # First component
            if variance_explained > 0.6:  # Explains >60% of variance
                if all_positive or all_negative:
                    return {
                        'factor_type': 'Market Factor',
                        'meaning': 'Overall Market Movement',
                        'explanation': f'This factor captures broad market risk that affects all assets similarly. When the market goes up/down, all your holdings tend to move in the same direction. This explains {variance_explained*100:.1f}% of your portfolio\'s volatility.'
                    }
                else:
                    return {
                        'factor_type': 'Market Factor', 
                        'meaning': 'General Market Exposure',
                        'explanation': f'This represents overall market movement, though some assets move opposite to others. This is your primary source of risk at {variance_explained*100:.1f}% of total volatility.'
                    }
            else:
                return {
                    'factor_type': 'Diversified Factor',
                    'meaning': 'Mixed Market Exposure', 
                    'explanation': f'Your portfolio is well-diversified - no single factor dominates. This factor explains {variance_explained*100:.1f}% of volatility.'
                }
        
        elif component_idx == 1:  # Second component
            if mixed_signs and unique_sectors > 1:
                return {
                    'factor_type': 'Sector/Style Factor',
                    'meaning': 'Growth vs Value / Sector Rotation',
                    'explanation': f'This factor captures differences between asset types - could be growth vs value stocks, or sector rotation effects. Explains {variance_explained*100:.1f}% of volatility beyond market movements.'
                }
            else:
                return {
                    'factor_type': 'Secondary Market Factor',
                    'meaning': 'Market Segment Movement',
                    'explanation': f'This represents a secondary market factor affecting specific segments of your portfolio. Contributes {variance_explained*100:.1f}% to total risk.'
                }
        
        elif component_idx == 2:  # Third component
            return {
                'factor_type': 'Idiosyncratic Factor',
                'meaning': 'Asset-Specific Risk',
                'explanation': f'This captures company or sector-specific risks that are independent of broad market movements. Represents {variance_explained*100:.1f}% of your portfolio volatility.'
            }
        
        else:  # Higher components
            return {
                'factor_type': 'Noise/Specific Risk',
                'meaning': 'Individual Asset Risk',
                'explanation': f'This represents very specific risks or noise in individual assets. Minor impact at {variance_explained*100:.1f}% of total volatility.'
            }

    def calculate_factor_etf_attribution(self, weights):
        """Perform factor attribution using Factor ETFs with regression analysis"""
        try:
            from sklearn.linear_model import Ridge, LinearRegression
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import r2_score
            
            if self.returns is None or not hasattr(self, 'factor_returns') or self.factor_returns is None:
                return None
            
            # Calculate portfolio returns
            portfolio_returns = (self.returns * weights).sum(axis=1)
            
            # Align dates between portfolio and factor returns
            common_dates = portfolio_returns.index.intersection(self.factor_returns.index)
            if len(common_dates) < 30:  # Need minimum observations
                return None
            
            portfolio_aligned = portfolio_returns.loc[common_dates]
            factor_aligned = self.factor_returns.loc[common_dates]
            
            # Standardize factor returns for better regression
            scaler = StandardScaler()
            factor_scaled = scaler.fit_transform(factor_aligned)
            
            # Try Ridge regression first (handles multicollinearity)
            ridge_reg = Ridge(alpha=0.1)
            ridge_reg.fit(factor_scaled, portfolio_aligned)
            ridge_pred = ridge_reg.predict(factor_scaled)
            ridge_r2 = r2_score(portfolio_aligned, ridge_pred)
            
            # Try OLS regression for comparison
            ols_reg = LinearRegression()
            ols_reg.fit(factor_scaled, portfolio_aligned)
            ols_pred = ols_reg.predict(factor_scaled)
            ols_r2 = r2_score(portfolio_aligned, ols_pred)
            
            # Choose better model
            if ridge_r2 > ols_r2:
                best_model = ridge_reg
                best_r2 = ridge_r2
                best_pred = ridge_pred
                model_type = "Ridge"
            else:
                best_model = ols_reg
                best_r2 = ols_r2
                best_pred = ols_pred
                model_type = "OLS"
            
            # Calculate factor exposures (betas)
            factor_exposures = {}
            factor_names = {
                'SPMO': 'Momentum',
                'RPV': 'Value', 
                'QVML': 'Multi Factor',
                'QUAL': 'Quality',
                'VFMF': 'Min Volatility'
            }
            
            # Transform coefficients back to original scale
            original_coeffs = best_model.coef_ / scaler.scale_
            
            for i, factor in enumerate(factor_aligned.columns):
                factor_name = factor_names.get(factor, factor)
                exposure = original_coeffs[i]
                factor_exposures[factor_name] = {
                    'beta': exposure,
                    'factor_etf': factor,
                    'contribution': abs(exposure) * factor_aligned[factor].std(),
                    'significance': 'High' if abs(exposure) > 0.3 else 'Medium' if abs(exposure) > 0.1 else 'Low'
                }
            
            # Calculate residual (unexplained) returns
            residuals = portfolio_aligned - best_pred
            residual_volatility = residuals.std()
            
            # Factor contribution analysis
            total_factor_risk = sum([exp['contribution'] for exp in factor_exposures.values()])
            
            for factor_name in factor_exposures:
                if total_factor_risk > 0:
                    factor_exposures[factor_name]['risk_contribution_pct'] = (
                        factor_exposures[factor_name]['contribution'] / total_factor_risk * 100
                    )
                else:
                    factor_exposures[factor_name]['risk_contribution_pct'] = 0
            
            # Portfolio alpha (intercept)
            alpha = best_model.intercept_ * 252  # Annualized
            
            attribution_results = {
                'factor_exposures': factor_exposures,
                'model_r_squared': best_r2,
                'model_type': model_type,
                'alpha_annualized': alpha,
                'residual_volatility': residual_volatility * np.sqrt(252),  # Annualized
                'systematic_risk_pct': best_r2 * 100,
                'idiosyncratic_risk_pct': (1 - best_r2) * 100,
                'factor_returns_used': list(factor_aligned.columns),
                'observation_count': len(common_dates),
                'portfolio_returns': portfolio_aligned,
                'factor_predictions': best_pred
            }
            
            # Add factor interpretation
            try:
                attribution_results['factor_insights'] = self._interpret_factor_etf_exposures(factor_exposures, best_r2)
            except Exception:
                attribution_results['factor_insights'] = {}
            
            return attribution_results
            
        except ImportError:
            if hasattr(st, 'error'):
                st.error("scikit-learn is required for Factor ETF attribution.")
            return None
        except Exception as e:
            if hasattr(st, 'warning'):
                st.warning(f"Factor ETF attribution failed: {str(e)}. Falling back to PCA analysis.")
            return None

    def _interpret_factor_etf_exposures(self, factor_exposures, r_squared):
        """Provide financial interpretation of factor exposures"""
        insights = {
            'dominant_factors': [],
            'portfolio_style': 'Balanced',
            'risk_concentration': 'Diversified',
            'actionable_recommendations': []
        }
        
        # Find dominant factors (high absolute exposure)
        sorted_factors = sorted(factor_exposures.items(), key=lambda x: abs(x[1]['beta']), reverse=True)
        
        for factor_name, exposure in sorted_factors[:3]:
            if abs(exposure['beta']) > 0.2:
                direction = "Overweight" if exposure['beta'] > 0 else "Underweight"
                insights['dominant_factors'].append({
                    'factor': factor_name,
                    'direction': direction,
                    'strength': exposure['significance'],
                    'beta': exposure['beta']
                })
        
        # Determine portfolio style
        momentum_exposure = factor_exposures.get('Momentum', {}).get('beta', 0)
        value_exposure = factor_exposures.get('Value', {}).get('beta', 0)
        quality_exposure = factor_exposures.get('Quality', {}).get('beta', 0)
        
        if momentum_exposure > 0.3:
            insights['portfolio_style'] = 'Growth/Momentum Focused'
        elif value_exposure > 0.3:
            insights['portfolio_style'] = 'Value Focused'
        elif quality_exposure > 0.3:
            insights['portfolio_style'] = 'Quality Focused'
        elif max(abs(momentum_exposure), abs(value_exposure), abs(quality_exposure)) < 0.15:
            insights['portfolio_style'] = 'Style Neutral'
        
        # Risk concentration analysis
        if r_squared > 0.85:
            insights['risk_concentration'] = 'Highly Systematic'
        elif r_squared > 0.7:
            insights['risk_concentration'] = 'Moderately Systematic'
        else:
            insights['risk_concentration'] = 'Diversified'
        
        # Generate actionable recommendations
        for factor_info in insights['dominant_factors']:
            factor_name = factor_info['factor']
            beta = factor_info['beta']
            if beta > 0.5:
                insights['actionable_recommendations'].append(
                    f"High {factor_name} exposure ({beta:.2f}) - consider reducing position size or hedging"
                )
            elif beta < -0.3:
                insights['actionable_recommendations'].append(
                    f"Negative {factor_name} exposure ({beta:.2f}) - portfolio moves opposite to this factor"
                )
        
        if r_squared < 0.6:
            insights['actionable_recommendations'].append(
                "Low factor R² suggests high stock-specific risk - consider broader diversification"
            )
        
        return insights

    def portfolio_performance(self, weights, return_type='sharpe'):
        """Calculate portfolio performance metrics"""
        if self.returns is None:
            self.calculate_returns()
        
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        
        if return_type == 'sharpe':
            return -portfolio_return / portfolio_std  # Negative for minimization
        elif return_type == 'return':
            return portfolio_return
        elif return_type == 'volatility':
            return portfolio_std
        else:
            return portfolio_return, portfolio_std

    def get_market_cap_weights(self):
        """Get market cap weights for the portfolio symbols"""
        try:
            market_caps = []
            for symbol in self.etf_symbols:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                market_cap = info.get('marketCap', 0)
                if market_cap == 0:
                    # Fallback: use shares outstanding * current price
                    shares = info.get('sharesOutstanding', 0)
                    price = info.get('currentPrice', 0) or info.get('regularMarketPrice', 0)
                    market_cap = shares * price if shares and price else 1e9  # Default to 1B if unavailable
                
                market_caps.append(market_cap)
            
            # Convert to weights
            total_market_cap = sum(market_caps)
            if total_market_cap > 0:
                weights = [mc / total_market_cap for mc in market_caps]
                return np.array(weights)
            else:
                # Equal weights fallback
                return np.array([1/len(self.etf_symbols)] * len(self.etf_symbols))
                
        except Exception:
            # Equal weights fallback
            return np.array([1/len(self.etf_symbols)] * len(self.etf_symbols))

    def black_litterman_optimization(self, market_cap_weights=None, views=None, view_confidences=None, 
                                   risk_aversion=3.0, tau=0.025, use_market_caps=True):
        """
        Black-Litterman portfolio optimization
        
        Parameters:
        - market_cap_weights: Prior market cap weights (if None, uses equal weights)
        - views: Dictionary of views {asset_index: expected_return}
        - view_confidences: Dictionary of confidence levels {asset_index: confidence_0_to_1}
        - risk_aversion: Risk aversion parameter (higher = more conservative)
        - tau: Uncertainty parameter (typically 0.01 to 0.05)
        """
        if self.returns is None:
            self.calculate_returns()
        
        num_assets = len(self.etf_symbols)
        
        # Use market cap weights as prior, or equal weights if not available
        if market_cap_weights is None and use_market_caps:
            w_market = self.get_market_cap_weights()
        elif market_cap_weights is not None:
            w_market = np.array(market_cap_weights)
            w_market = w_market / w_market.sum()  # Normalize
        else:
            w_market = np.array([1/num_assets] * num_assets)
        
        # Annualized covariance matrix
        cov_annual = self.cov_matrix * 252
        
        # Implied equilibrium returns (CAPM)
        pi = risk_aversion * np.dot(cov_annual, w_market)
        
        # If no views provided, return market portfolio
        if views is None or len(views) == 0:
            return w_market
        
        # Construct P matrix (picking matrix) and Q vector (views)
        P = np.zeros((len(views), num_assets))
        Q = np.zeros(len(views))
        
        for i, (asset_idx, view_return) in enumerate(views.items()):
            P[i, asset_idx] = 1.0
            Q[i] = view_return
        
        # Construct Omega matrix (uncertainty of views)
        Omega = np.zeros((len(views), len(views)))
        for i, (asset_idx, confidence) in enumerate(view_confidences.items()):
            # Lower confidence = higher uncertainty
            view_uncertainty = tau * P[i, :].T @ cov_annual @ P[i, :] / confidence
            Omega[i, i] = view_uncertainty
        
        # Black-Litterman formula
        try:
            # M1: Updated expected returns
            tau_cov = tau * cov_annual
            inv_tau_cov = np.linalg.inv(tau_cov)
            inv_omega = np.linalg.inv(Omega)
            
            # New expected returns
            middle_term = np.linalg.inv(inv_tau_cov + P.T @ inv_omega @ P)
            mu_bl = middle_term @ (inv_tau_cov @ pi + P.T @ inv_omega @ Q)
            
            # M2: Updated covariance matrix
            cov_bl = middle_term
            
            # Optimal portfolio weights
            inv_cov_bl = np.linalg.inv(cov_bl)
            w_optimal = inv_cov_bl @ mu_bl / risk_aversion
            
            # Normalize weights to sum to 1
            w_optimal = w_optimal / w_optimal.sum()
            
            # Ensure no negative weights (long-only constraint)
            w_optimal = np.maximum(w_optimal, 0)
            w_optimal = w_optimal / w_optimal.sum()
            
            return w_optimal
            
        except np.linalg.LinAlgError:
            st.warning("Black-Litterman optimization failed due to matrix inversion issues. Using market weights.")
            return w_market

    def optimize_portfolio(self, objective='sharpe', bl_params=None):
        """Optimize portfolio based on objective"""
        if self.returns is None:
            self.calculate_returns()
        
        num_assets = len(self.etf_symbols)
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        if objective == 'equal_weighted':
            # Equal weighted portfolio
            optimal_weights = np.array([1/num_assets] * num_assets)
        elif objective == 'black_litterman':
            # Black-Litterman optimization
            if bl_params is None:
                bl_params = {}
            optimal_weights = self.black_litterman_optimization(**bl_params)
        elif objective == 'sharpe':
            # Maximize Sharpe ratio
            result = minimize(self.portfolio_performance, 
                            np.array([1/num_assets] * num_assets),
                            method='SLSQP', bounds=bounds, constraints=constraints,
                            args=('sharpe',))
            optimal_weights = result.x
        elif objective == 'minimum_variance':
            # Minimize variance
            result = minimize(self.portfolio_performance, 
                            np.array([1/num_assets] * num_assets),
                            method='SLSQP', bounds=bounds, constraints=constraints,
                            args=('volatility',))
            optimal_weights = result.x
        elif objective == 'drawdown':
            # Minimize maximum drawdown
            optimal_weights = self._optimize_drawdown()
        elif objective == 'risk_adjusted_drawdown':
            # Risk-adjusted drawdown optimization
            optimal_weights = self._optimize_risk_adjusted_drawdown()
        else:
            # Default to equal weighted
            optimal_weights = np.array([1/num_assets] * num_assets)
        
        return optimal_weights

    def _optimize_drawdown(self):
        """Optimize for minimum maximum drawdown"""
        num_assets = len(self.etf_symbols)
        
        def max_drawdown_objective(weights):
            portfolio_returns = (self.returns * weights).sum(axis=1)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns / running_max) - 1
            return -drawdown.min()  # Minimize maximum drawdown
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        result = minimize(max_drawdown_objective, 
                        np.array([1/num_assets] * num_assets),
                        method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else np.array([1/num_assets] * num_assets)

    def _optimize_risk_adjusted_drawdown(self):
        """Optimize for risk-adjusted drawdown"""
        num_assets = len(self.etf_symbols)
        
        def risk_adjusted_drawdown_objective(weights):
            portfolio_returns = (self.returns * weights).sum(axis=1)
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns / running_max) - 1
            max_dd = -drawdown.min()
            volatility = portfolio_returns.std() * np.sqrt(252)
            return max_dd / volatility if volatility > 0 else float('inf')
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        result = minimize(risk_adjusted_drawdown_objective, 
                        np.array([1/num_assets] * num_assets),
                        method='SLSQP', bounds=bounds, constraints=constraints)
        
        return result.x if result.success else np.array([1/num_assets] * num_assets)

    def calculate_portfolio_metrics(self, weights):
        """Calculate comprehensive portfolio metrics"""
        if self.returns is None:
            self.calculate_returns()
        
        # Portfolio returns
        portfolio_returns = (self.returns * weights).sum(axis=1)
        
        # Basic metrics
        annual_return = portfolio_returns.mean() * 252
        annual_volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Drawdown analysis
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()
        
        # VaR and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_std if downside_std > 0 else 0
        
        metrics = {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'sortino_ratio': sortino_ratio,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns - 1,
            'drawdown': drawdown
        }
        
        return metrics

    def fetch_iv_data(self, tenor_days=30, show_progress=True):
        """Fetch implied volatility data for symbols using AlphaQuery IV scraper"""
        if not self.iv_scraper:
            return None
        
        try:
            # Filter for equity symbols (ETFs typically don't have liquid options)
            equity_symbols = []
            etf_patterns = ['SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'EFA', 'EEM',
                           'BND', 'TLT', 'HYG', 'LQD', 'GLD', 'SLV', 'USO', 'XLF', 'XLK',
                           'XLE', 'XLI', 'XLV', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']
            
            for symbol in self.etf_symbols:
                # Check if it's likely a stock (not an ETF)
                if symbol not in etf_patterns and len(symbol) <= 5 and not any(x in symbol for x in ['ETF', 'FUND']):
                    equity_symbols.append(symbol)
            
            if not equity_symbols:
                if show_progress:
                    st.info("No equity symbols found for IV analysis. IV data is only available for individual stocks.")
                return None
            
            if show_progress:
                st.info(f"Fetching {tenor_days}-day IV data for {len(equity_symbols)} equity symbols...")
            
            iv_results = {}
            if show_progress:
                progress_bar = st.progress(0)
            
            for i, symbol in enumerate(equity_symbols):
                try:
                    if show_progress:
                        progress_bar.progress((i + 1) / len(equity_symbols))
                    
                    # Fetch IV data for the symbol
                    iv_data = self.iv_scraper.get_iv_data(symbol, tenor_days=tenor_days)
                    
                    if iv_data and 'current_iv' in iv_data:
                        # Extract IV data from the actual response format
                        current_iv = iv_data.get('current_iv', 0)
                        hist_vol = iv_data.get('historical_volatility', 0)
                        volume = iv_data.get('volume', 0)
                        open_interest = iv_data.get('open_interest', 0)
                        
                        iv_results[symbol] = {
                            'symbol': symbol,
                            'tenor_days': tenor_days,
                            'avg_iv': current_iv * 100,  # Convert to percentage
                            'call_iv': current_iv * 100,  # Use current IV for both (API doesn't separate)
                            'put_iv': current_iv * 100,   # Use current IV for both
                            'historical_vol': hist_vol * 100,
                            'volume': volume * 100,
                            'open_interest': open_interest * 100,
                            'iv_vs_hv': (current_iv - hist_vol) * 100,  # IV vs Historical Vol difference
                            'data_points': 1,  # Single data point from API
                            'date': iv_data.get('date', 'N/A')
                        }
                        
                        if show_progress:
                            st.success(f"{symbol}: IV data fetched successfully")
                    else:
                        if show_progress:
                            st.warning(f"{symbol}: No IV data available")
                        
                except Exception as e:
                    if show_progress:
                        st.warning(f"{symbol}: Error fetching IV data - {str(e)}")
                    continue
            
            if show_progress and 'progress_bar' in locals():
                progress_bar.empty()
                
            self.iv_data = iv_results
            return iv_results
            
        except Exception as e:
            if show_progress:
                st.error(f"Error in IV data fetching: {str(e)}")
            return None

    def fetch_symbol_info(self):
        """Fetch comprehensive information for all symbols using yfinance"""
        try:
            symbol_info = {}
            
            for symbol in self.etf_symbols:
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    # Get basic info
                    quote_type = info.get('quoteType', 'N/A')
                    
                    # Fix sector for ETFs
                    sector = info.get('sector', 'N/A')
                    if sector == 'N/A' and quote_type == 'ETF':
                        sector = 'ETF'
                    
                    symbol_info[symbol] = {
                        'symbol': symbol,
                        'longName': info.get('longName', symbol),
                        'shortName': info.get('shortName', symbol),
                        'sector': sector,
                        'industry': info.get('industry', 'N/A'),
                        'marketCap': info.get('marketCap', 0),
                        'enterpriseValue': info.get('enterpriseValue', 0),
                        'trailingPE': info.get('trailingPE', 0),
                        'forwardPE': info.get('forwardPE', 0),
                        'pegRatio': info.get('pegRatio', 0),
                        'priceToBook': info.get('priceToBook', 0),
                        'debtToEquity': info.get('debtToEquity', 0),
                        'returnOnEquity': info.get('returnOnEquity', 0),
                        'beta': info.get('beta', 0),
                        'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
                        'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 0),
                        # Enhanced dividend yield calculation
                        'trailingAnnualDividendYield': info.get('trailingAnnualDividendYield', 0),
                        'dividendYield': info.get('dividendYield', 0),
                        'yieldPct': info.get('yield', 0),  # ETF yield field
                        'payoutRatio': info.get('payoutRatio', 0),
                        'volume': info.get('volume', 0),
                        'averageVolume': info.get('averageVolume', 0),
                        'currency': info.get('currency', 'USD'),
                        'exchange': info.get('fullExchangeName', info.get('exchange', 'N/A')),
                        'quoteType': info.get('quoteType', 'N/A'),
                        'currentPrice': info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0),
                        'dividend_yield': 0,  # Will be calculated below
                        # ETF specific fields
                        'totalAssets': info.get('totalAssets', 0),
                        'expenseRatio': info.get('expenseRatio', 0),
                        'fundFamily': info.get('fundFamily', 'N/A'),
                        'category': info.get('category', 'N/A'),
                        'navPrice': info.get('navPrice', 0),
                        # Additional financial metrics
                        'revenueGrowth': info.get('revenueGrowth', 0),
                        'earningsGrowth': info.get('earningsGrowth', 0),
                        'profitMargins': info.get('profitMargins', 0),
                        'operatingMargins': info.get('operatingMargins', 0),
                        'returnOnAssets': info.get('returnOnAssets', 0),
                        'currentRatio': info.get('currentRatio', 0),
                        'quickRatio': info.get('quickRatio', 0),
                        # Analyst data
                        'targetMeanPrice': info.get('targetMeanPrice', 0),
                        'targetHighPrice': info.get('targetHighPrice', 0),
                        'targetLowPrice': info.get('targetLowPrice', 0),
                        'recommendationMean': info.get('recommendationMean', 0),
                        'recommendationKey': info.get('recommendationKey', 'N/A'),
                        'numberOfAnalystOpinions': info.get('numberOfAnalystOpinions', 0),
                        # Short interest
                        'sharesOutstanding': info.get('sharesOutstanding', 0),
                        'floatShares': info.get('floatShares', 0),
                        'sharesShort': info.get('sharesShort', 0),
                        'shortRatio': info.get('shortRatio', 0),
                        'shortPercentOfFloat': info.get('shortPercentOfFloat', 0)
                    }
                    
                    # Fetch institutional holders data
                    try:
                        institutional_holders = ticker.institutional_holders
                        if institutional_holders is not None and not institutional_holders.empty:
                            top_holders = institutional_holders.head(3)['Holder'].tolist()
                            symbol_info[symbol]['topInstitutionalHolders'] = ', '.join(top_holders)
                        else:
                            symbol_info[symbol]['topInstitutionalHolders'] = 'N/A'
                    except:
                        symbol_info[symbol]['topInstitutionalHolders'] = 'N/A'
                    
                    # Fetch recommendations data
                    try:
                        recommendations = ticker.recommendations
                        if recommendations is not None and not recommendations.empty:
                            latest_rec = recommendations.tail(1)
                            strong_buy = latest_rec['strongBuy'].iloc[0] if 'strongBuy' in latest_rec.columns else 0
                            buy = latest_rec['buy'].iloc[0] if 'buy' in latest_rec.columns else 0
                            hold = latest_rec['hold'].iloc[0] if 'hold' in latest_rec.columns else 0
                            sell = latest_rec['sell'].iloc[0] if 'sell' in latest_rec.columns else 0
                            symbol_info[symbol]['analystRecommendations'] = f"Buy: {strong_buy + buy}, Hold: {hold}, Sell: {sell}"
                        else:
                            symbol_info[symbol]['analystRecommendations'] = 'N/A'
                    except:
                        symbol_info[symbol]['analystRecommendations'] = 'N/A'
                    
                    # Enhanced dividend yield calculation
                    try:
                        # Try multiple dividend yield fields
                        dividend_yield = 0
                        trailing_yield = info.get('trailingAnnualDividendYield', 0)
                        regular_yield = info.get('dividendYield', 0)
                        etf_yield = info.get('yield', 0)
                        
                        # Convert regular_yield from percentage to decimal if > 10 (likely percentage format)
                        if regular_yield and regular_yield > 10:
                            regular_yield = regular_yield / 100
                        
                        # Use the highest non-zero value (now all in decimal format)
                        for yield_val in [trailing_yield, regular_yield, etf_yield]:
                            if yield_val and yield_val > dividend_yield:
                                dividend_yield = yield_val
                        
                        # If still no dividend yield, try to calculate from dividends history
                        if dividend_yield == 0:
                            try:
                                dividends = ticker.dividends
                                current_price = info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose', 0)
                                
                                if not dividends.empty and current_price > 0:
                                    # Get last 12 months of dividends
                                    recent_dividends = dividends.last('365D')
                                    if not recent_dividends.empty:
                                        annual_dividends = recent_dividends.sum()
                                        dividend_yield = annual_dividends / current_price
                                        
                            except:
                                pass
                        
                        symbol_info[symbol]['calculatedDividendYield'] = dividend_yield
                        symbol_info[symbol]['dividend_yield'] = dividend_yield
                        
                    except:
                        symbol_info[symbol]['calculatedDividendYield'] = 0
                        symbol_info[symbol]['dividend_yield'] = 0
                    
                except Exception as e:
                    symbol_info[symbol] = {
                        'symbol': symbol,
                        'error': str(e),
                        'longName': symbol,
                        'shortName': symbol
                    }
                    continue
            
            return symbol_info
            
        except Exception as e:
            st.error(f"Error fetching symbol information: {str(e)}")
            return {}

def create_allocation_chart(symbols, weights):
    """Create portfolio allocation pie chart"""
    fig = go.Figure(data=[go.Pie(
        labels=symbols,
        values=weights,
        hole=0.3,
        textinfo='label+percent',
        textposition='auto',
        marker=dict(
            colors=['#1976d2', '#2196f3', '#42a5f5', '#64b5f6', '#90caf9', '#bbdefb', '#e3f2fd', '#f3e5f5']
        )
    )])
    
    fig.update_layout(
        title="Portfolio Allocation",
        font=dict(size=12),
        showlegend=True,
        height=400
    )
    
    return fig

def create_performance_chart(metrics):
    """Create performance charts"""
    # Cumulative returns chart
    cum_returns_fig = go.Figure()
    cum_returns_fig.add_trace(go.Scatter(
        x=metrics['cumulative_returns'].index,
        y=metrics['cumulative_returns'] * 100,
        mode='lines',
        name='Portfolio',
        line=dict(color='#1976d2', width=3)
    ))
    
    cum_returns_fig.update_layout(
        title="Cumulative Returns (%)",
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode='x',
        height=400
    )
    
    # Drawdown chart
    drawdown_fig = go.Figure()
    drawdown_fig.add_trace(go.Scatter(
        x=metrics['drawdown'].index,
        y=metrics['drawdown'] * 100,
        mode='lines',
        fill='tozeroy',
        name='Drawdown',
        line=dict(color='#d32f2f', width=2)
    ))
    
    drawdown_fig.update_layout(
        title="Portfolio Drawdown (%)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x',
        height=400
    )
    
    return cum_returns_fig, drawdown_fig

def main():
    # Main header
    st.markdown('<h1 class="main-title">Portfolio Optimization Tool</h1>', unsafe_allow_html=True)
    st.info("Modern Portfolio Theory (MPT) optimization tool for ETFs and stocks")
    
    # Sidebar for inputs
    st.sidebar.header("Settings")
    
    # ETF/Stock input
    ticker_input = st.sidebar.text_input(
        "ETF/Stock Symbols (comma-separated)",
        value="AAPL,GOOGL,MSFT,NVDA,TSLA,SPY,QQQ",
        help="Enter Yahoo Finance symbols separated by commas. Mix stocks and ETFs for IV analysis on stocks."
    )
    
    # Optimization objective
    objective = st.sidebar.selectbox(
        "Optimization Objective",
        ["equal_weighted", "sharpe", "minimum_variance", "drawdown", "risk_adjusted_drawdown", "black_litterman"],
        format_func=lambda x: {
            "equal_weighted": "Equal Weighted Portfolio",
            "sharpe": "Maximize Sharpe Ratio",
            "minimum_variance": "Minimum Variance Portfolio",
            "drawdown": "Minimize Drawdown",
            "risk_adjusted_drawdown": "Risk-Adjusted Drawdown",
            "black_litterman": "Black-Litterman Model"
        }[x]
    )
    
    # Data period
    period = st.sidebar.selectbox(
        "Data Period",
        ["6mo", "1y", "2y", "5y", "max"],
        index=2,
        format_func=lambda x: {
            "6mo": "6 Months",
            "1y": "1 Year", 
            "2y": "2 Years",
            "5y": "5 Years",
            "max": "Maximum Available"
        }[x]
    )
    
    # Black-Litterman specific controls
    bl_params = {}
    if objective == "black_litterman":
        st.sidebar.subheader("Black-Litterman Parameters")
        
        # Risk aversion
        risk_aversion = st.sidebar.slider(
            "Risk Aversion", 
            min_value=1.0, 
            max_value=10.0, 
            value=3.0, 
            step=0.5,
            help="Higher values = more conservative portfolio"
        )
        
        # Tau parameter
        tau = st.sidebar.slider(
            "Tau (Uncertainty)", 
            min_value=0.01, 
            max_value=0.1, 
            value=0.025, 
            step=0.005,
            help="Market uncertainty parameter (0.01-0.05 typical)"
        )
        
        # Market cap weighting option
        use_market_caps = st.sidebar.checkbox(
            "Use Market Cap Weights as Prior",
            value=True,
            help="Use market capitalization to weight the equilibrium portfolio (recommended)"
        )
        
        # Views input
        st.sidebar.subheader("Market Views (Optional)")
        st.sidebar.write("Express your views on expected returns:")
        
        # Parse symbols first to show in views
        temp_symbols = [s.strip().upper() for s in ticker_input.split(',') if s.strip()]
        views = {}
        view_confidences = {}
        
        if len(temp_symbols) > 0:
            for i, symbol in enumerate(temp_symbols):
                col1, col2 = st.sidebar.columns([1, 1])
                
                with col1:
                    view_return = st.number_input(
                        f"{symbol} Expected Return (%)",
                        min_value=-50.0,
                        max_value=100.0,
                        value=0.0,
                        step=1.0,
                        key=f"view_{symbol}"
                    )
                
                with col2:
                    confidence = st.slider(
                        f"{symbol} Confidence",
                        min_value=0.1,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                        key=f"conf_{symbol}",
                        help="How confident are you? (0.1=low, 1.0=very high)"
                    )
                
                if view_return != 0.0:
                    views[i] = view_return / 100  # Convert percentage to decimal
                    view_confidences[i] = confidence
        
        bl_params = {
            'risk_aversion': risk_aversion,
            'tau': tau,
            'views': views if views else None,
            'view_confidences': view_confidences if view_confidences else None,
            'use_market_caps': use_market_caps
        }
        
        if views:
            st.sidebar.success(f"{len(views)} market views specified")
        else:
            st.sidebar.info("No views specified - will use market portfolio")
    
    # Optimize button
    optimize_button = st.sidebar.button("Optimize Portfolio", type="primary")
    
    if optimize_button and ticker_input:
        # Parse symbols
        symbols = [s.strip().upper() for s in ticker_input.split(',') if s.strip()]
        
        if len(symbols) < 2:
            st.error("Please enter at least 2 symbols for optimization")
            return
        
        # Initialize optimizer
        with st.spinner("Fetching data and optimizing portfolio..."):
            optimizer = ETFPortfolioOptimizer(symbols)
            
            try:
                # Fetch data
                data = optimizer.fetch_data(period=period)
                
                # Calculate returns
                optimizer.calculate_returns()
                
                # Fetch benchmark data (for future use)
                optimizer.fetch_benchmark_data(period=period)
                
                # Fetch factor ETF data for correlation analysis
                optimizer.fetch_factor_correlation(period=period)
                
                # Fetch symbol information (factsheet data)
                st.info("Fetching symbol information...")
                symbol_info = optimizer.fetch_symbol_info()
                
                # Auto-fetch IV data for equity symbols (non-intrusive)
                iv_data = optimizer.fetch_iv_data(tenor_days=30, show_progress=False)
                
                # Optimize portfolio
                if objective == "black_litterman":
                    optimal_weights = optimizer.optimize_portfolio(objective=objective, bl_params=bl_params)
                else:
                    optimal_weights = optimizer.optimize_portfolio(objective=objective)
                
                # Calculate metrics
                metrics = optimizer.calculate_portfolio_metrics(optimal_weights)
                
                # Display results
                st.header("Optimization Results")
                
                # Show Black-Litterman specific information
                if objective == "black_litterman":
                    st.subheader("Black-Litterman Model Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Risk Aversion", f"{bl_params['risk_aversion']:.1f}")
                    with col2:
                        st.metric("Tau Parameter", f"{bl_params['tau']:.3f}")
                    with col3:
                        views_count = len(bl_params['views']) if bl_params['views'] else 0
                        st.metric("Market Views", views_count)
                    
                    if bl_params['views']:
                        st.subheader("Your Market Views Applied")
                        views_df = pd.DataFrame([
                            {
                                'Asset': symbols[asset_idx],
                                'Expected Return': f"{view_return*100:.1f}%",
                                'Confidence': f"{bl_params['view_confidences'][asset_idx]*100:.0f}%"
                            }
                            for asset_idx, view_return in bl_params['views'].items()
                        ])
                        st.dataframe(views_df, use_container_width=True)
                        
                        st.info("**Black-Litterman Insight**: The model combines your views with market equilibrium to create a more robust optimization that accounts for uncertainty in your predictions.")
                    else:
                        st.info("**Market Portfolio**: No views specified, so the model returned market-cap weighted portfolio (or equal weights if market caps unavailable).")
                
                    # Black-Litterman Interpretation Guide
                    with st.expander("How to Interpret Black-Litterman Results", expanded=False):
                        st.markdown("""
                        ### Understanding Your Black-Litterman Portfolio
                        
                        **What the Model Did:**
                        1. **Started with Market Equilibrium**: Used current market cap weights as the "neutral" baseline
                        2. **Incorporated Your Views**: Blended your specific predictions with market consensus
                        3. **Accounted for Uncertainty**: Adjusted for how confident you are in each view
                        4. **Optimized Allocation**: Found the optimal balance between risk and expected return
                        
                        **How to Read the Results:**
                        
                        **Weight Changes from Market:**
                        - **Higher weights** than market cap = Model agrees with your bullish views
                        - **Lower weights** than market cap = Model is more cautious or disagrees
                        - **Similar weights** = Your views align with market expectations
                        
                        **Impact of Your Views:**
                        - **High confidence views** (80-100%) = Model heavily weights your opinion
                        - **Low confidence views** (20-40%) = Model stays closer to market consensus
                        - **Extreme allocations** suggest strong conviction in your views
                        
                        **Risk vs. Views Trade-off:**
                        - **Higher risk aversion** = More conservative, closer to market portfolio
                        - **Lower risk aversion** = More aggressive, follows your views more closely
                        - **Higher tau** = More uncertainty, stays closer to market baseline
                        
                        **Practical Interpretation:**
                        
                        **Example Scenarios:**
                        - **Tech stock gets 25% weight** (vs 15% market cap) → Your bullish tech view is reflected
                        - **Bond ETF gets 5% weight** (vs 20% market cap) → Model thinks your growth views favor equities
                        - **Equal to market weights** → Your views don't significantly differ from market consensus
                        
                        **Warning Signs:**
                        - **Extreme concentrations** (>50% in one asset) → Consider moderating views or confidence
                        - **Zero weights** → Model strongly disagrees with that asset given your other views
                        - **Highly volatile changes** with small view adjustments → Consider lower confidence levels
                        
                        **Action Items:**
                        1. **Compare to market cap weights** to see where your views matter most
                        2. **Test sensitivity** by adjusting confidence levels
                        3. **Consider rebalancing triggers** based on how views change over time
                        4. **Monitor performance** of view-driven allocations vs. market weights
                        
                        **Pro Tips:**
                        - Start with **moderate confidence** (40-60%) until you validate the model
                        - Use **higher risk aversion** (5-8) for more conservative portfolios
                        - **Diversify your views** across different asset classes
                        - **Update views regularly** as market conditions change
                        """)
                
                    st.markdown("---")
                
                # Portfolio allocation
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.plotly_chart(create_allocation_chart(symbols, optimal_weights), use_container_width=True)
                
                with col2:
                    st.subheader("Portfolio Weights")
                    weights_df = pd.DataFrame({
                        'Symbol': symbols,
                        'Weight (%)': [w * 100 for w in optimal_weights]
                    })
                    st.dataframe(weights_df, use_container_width=True)
                
                # Performance metrics
                st.header("Performance Metrics")
                
                metric_cols = st.columns(4)
                with metric_cols[0]:
                    st.metric("Annual Return", f"{metrics['annual_return']*100:.2f}%")
                
                with metric_cols[1]:
                    st.metric("Annual Volatility", f"{metrics['annual_volatility']*100:.2f}%")
                
                with metric_cols[2]:
                    st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
                
                with metric_cols[3]:
                    st.metric("Max Drawdown", f"{metrics['max_drawdown']*100:.2f}%")
                
                # Additional metrics
                additional_cols = st.columns(3)
                with additional_cols[0]:
                    st.metric("Calmar Ratio", f"{metrics['calmar_ratio']:.3f}")
                
                with additional_cols[1]:
                    st.metric("Sortino Ratio", f"{metrics['sortino_ratio']:.3f}")
                
                with additional_cols[2]:
                    st.metric("VaR (95%)", f"{metrics['var_95']*100:.2f}%")
                
                # Performance charts
                st.header("Performance Charts")
                
                cum_chart, dd_chart = create_performance_chart(metrics)
                
                chart_col1, chart_col2 = st.columns([1, 1])
                with chart_col1:
                    st.plotly_chart(cum_chart, use_container_width=True)
                
                with chart_col2:
                    st.plotly_chart(dd_chart, use_container_width=True)
                
                # Portfolio Factsheet
                st.header("Portfolio Factsheet")
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["Basic Info", "Financial Metrics", "Analyst Data", "Advanced Data"])
                
                with tab1:
                    # Basic portfolio information
                    basic_data = []
                    for symbol in symbols:
                        info = symbol_info.get(symbol, {})
                        weight = optimal_weights[symbols.index(symbol)] * 100
                        iv_info = iv_data.get(symbol, {}) if iv_data else {}
                        
                        basic_data.append({
                            'Symbol': symbol,
                            'Name': info.get('longName', symbol)[:40] + '...' if len(info.get('longName', symbol)) > 40 else info.get('longName', symbol),
                            'Weight (%)': f"{weight:.2f}%",
                            'Sector': info.get('sector', 'N/A'),
                            'Market Cap': f"${info.get('marketCap', 0)/1e9:.1f}B" if info.get('marketCap', 0) > 0 else 'N/A',
                            'Current Price': f"${info.get('currentPrice', 0):.2f}" if info.get('currentPrice') else 'N/A',
                            'Exchange': info.get('exchange', 'N/A'),
                            'IV (30d)': f"{iv_info.get('avg_iv', 0):.1f}%" if iv_info.get('avg_iv') else 'N/A'
                        })
                    
                    basic_df = pd.DataFrame(basic_data)
                    st.dataframe(basic_df, use_container_width=True)
                
                with tab2:
                    # Financial metrics
                    financial_data = []
                    for symbol in symbols:
                        info = symbol_info.get(symbol, {})
                        weight = optimal_weights[symbols.index(symbol)] * 100
                        
                        financial_data.append({
                            'Symbol': symbol,
                            'Weight (%)': f"{weight:.2f}%",
                            'P/E Ratio': f"{info.get('trailingPE', 0):.2f}" if info.get('trailingPE') else 'N/A',
                            'Beta': f"{info.get('beta', 0):.2f}" if info.get('beta') else 'N/A',
                            'Profit Margin': f"{info.get('profitMargins', 0)*100:.2f}%" if info.get('profitMargins') else 'N/A',
                            'Operating Margin': f"{info.get('operatingMargins', 0)*100:.2f}%" if info.get('operatingMargins') else 'N/A',
                            'ROA': f"{info.get('returnOnAssets', 0)*100:.2f}%" if info.get('returnOnAssets') else 'N/A',
                            'ROE': f"{info.get('returnOnEquity', 0)*100:.2f}%" if info.get('returnOnEquity') else 'N/A',
                            'Current Ratio': f"{info.get('currentRatio', 0):.2f}" if info.get('currentRatio') else 'N/A',
                            'Dividend Yield': f"{info.get('calculatedDividendYield', 0)*100:.2f}%" if info.get('calculatedDividendYield') else 'N/A'
                        })
                    
                    financial_df = pd.DataFrame(financial_data)
                    st.dataframe(financial_df, use_container_width=True)
                
                with tab3:
                    # Analyst data
                    analyst_data = []
                    for symbol in symbols:
                        info = symbol_info.get(symbol, {})
                        weight = optimal_weights[symbols.index(symbol)] * 100
                        current_price = info.get('currentPrice', 0)
                        target_price = info.get('targetMeanPrice', 0)
                        upside = ((target_price - current_price) / current_price * 100) if current_price > 0 and target_price > 0 else 0
                        
                        analyst_data.append({
                            'Symbol': symbol,
                            'Weight (%)': f"{weight:.2f}%",
                            'Current Price': f"${current_price:.2f}" if current_price else 'N/A',
                            'Target Price': f"${target_price:.2f}" if target_price else 'N/A',
                            'Upside/Downside': f"{upside:+.1f}%" if upside != 0 else 'N/A',
                            'Recommendation': info.get('recommendationKey', 'N/A').title(),
                            'Analyst Count': info.get('numberOfAnalystOpinions', 0),
                            'Recommendations': info.get('analystRecommendations', 'N/A')
                        })
                    
                    analyst_df = pd.DataFrame(analyst_data)
                    st.dataframe(analyst_df, use_container_width=True)
                
                with tab4:
                    # Advanced data (short interest, institutional holdings, growth)
                    advanced_data = []
                    for symbol in symbols:
                        info = symbol_info.get(symbol, {})
                        weight = optimal_weights[symbols.index(symbol)] * 100
                        
                        advanced_data.append({
                            'Symbol': symbol,
                            'Weight (%)': f"{weight:.2f}%",
                            'Revenue Growth': f"{info.get('revenueGrowth', 0)*100:.2f}%" if info.get('revenueGrowth') else 'N/A',
                            'Earnings Growth': f"{info.get('earningsGrowth', 0)*100:.2f}%" if info.get('earningsGrowth') else 'N/A',
                            'Short Ratio': f"{info.get('shortRatio', 0):.2f}" if info.get('shortRatio') else 'N/A',
                            'Short % Float': f"{info.get('shortPercentOfFloat', 0)*100:.2f}%" if info.get('shortPercentOfFloat') else 'N/A',
                            'Float Shares': f"{info.get('floatShares', 0)/1e6:.1f}M" if info.get('floatShares', 0) > 0 else 'N/A',
                            'Top Institutions': info.get('topInstitutionalHolders', 'N/A')[:50] + '...' if len(info.get('topInstitutionalHolders', '')) > 50 else info.get('topInstitutionalHolders', 'N/A')
                        })
                    
                    advanced_df = pd.DataFrame(advanced_data)
                    st.dataframe(advanced_df, use_container_width=True)
                
                # Enhanced Portfolio Analytics
                st.header("Portfolio Analytics")
                
                # Calculate weighted averages
                def calc_weighted_avg(field, weights, symbol_infos):
                    total_weight = 0
                    weighted_sum = 0
                    for i, symbol in enumerate(symbols):
                        info = symbol_infos.get(symbol, {})
                        value = info.get(field, 0)
                        if value and value > 0:
                            weight = weights[i]
                            weighted_sum += value * weight
                            total_weight += weight
                    return weighted_sum / total_weight if total_weight > 0 else 0
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.subheader("Valuation Metrics")
                    weighted_pe = calc_weighted_avg('trailingPE', optimal_weights, symbol_info)
                    weighted_pb = calc_weighted_avg('priceToBook', optimal_weights, symbol_info)
                    weighted_beta = calc_weighted_avg('beta', optimal_weights, symbol_info)
                    
                    st.metric("Weighted P/E", f"{weighted_pe:.2f}" if weighted_pe > 0 else "N/A")
                    st.metric("Weighted P/B", f"{weighted_pb:.2f}" if weighted_pb > 0 else "N/A")
                    st.metric("Weighted Beta", f"{weighted_beta:.2f}" if weighted_beta > 0 else "N/A")
                
                with col2:
                    st.subheader("Profitability")
                    weighted_profit_margin = calc_weighted_avg('profitMargins', optimal_weights, symbol_info)
                    weighted_roe = calc_weighted_avg('returnOnEquity', optimal_weights, symbol_info)
                    weighted_roa = calc_weighted_avg('returnOnAssets', optimal_weights, symbol_info)
                    
                    st.metric("Weighted Profit Margin", f"{weighted_profit_margin*100:.2f}%" if weighted_profit_margin > 0 else "N/A")
                    st.metric("Weighted ROE", f"{weighted_roe*100:.2f}%" if weighted_roe > 0 else "N/A")
                    st.metric("Weighted ROA", f"{weighted_roa*100:.2f}%" if weighted_roa > 0 else "N/A")
                
                with col3:
                    st.subheader("Growth Metrics")
                    weighted_rev_growth = calc_weighted_avg('revenueGrowth', optimal_weights, symbol_info)
                    weighted_earnings_growth = calc_weighted_avg('earningsGrowth', optimal_weights, symbol_info)
                    avg_target_upside = np.mean([
                        ((info.get('targetMeanPrice', 0) - info.get('currentPrice', 0)) / info.get('currentPrice', 1) * 100)
                        for info in symbol_info.values()
                        if info.get('targetMeanPrice', 0) > 0 and info.get('currentPrice', 0) > 0
                    ])
                    
                    st.metric("Weighted Revenue Growth", f"{weighted_rev_growth*100:.2f}%" if weighted_rev_growth > 0 else "N/A")
                    st.metric("Weighted Earnings Growth", f"{weighted_earnings_growth*100:.2f}%" if weighted_earnings_growth > 0 else "N/A")
                    st.metric("Avg Analyst Upside", f"{avg_target_upside:+.1f}%" if not np.isnan(avg_target_upside) else "N/A")
                
                with col4:
                    st.subheader("Portfolio Composition")
                    total_market_cap = sum([info.get('marketCap', 0) * optimal_weights[i] for i, info in enumerate(symbol_info.values())])
                    
                    # Sector breakdown
                    sector_weights = {}
                    for i, symbol in enumerate(symbols):
                        sector = symbol_info.get(symbol, {}).get('sector', 'Other')
                        weight = optimal_weights[i] * 100
                        sector_weights[sector] = sector_weights.get(sector, 0) + weight
                    
                    top_sector = max(sector_weights.items(), key=lambda x: x[1]) if sector_weights else ('N/A', 0)
                    
                    st.metric("Weighted Market Cap", f"${total_market_cap/1e12:.2f}T" if total_market_cap > 0 else "N/A")
                    st.metric("Largest Sector", f"{top_sector[0]}")
                    st.metric("Sector Weight", f"{top_sector[1]:.1f}%")
                
                # Sector allocation chart
                if sector_weights:
                    st.subheader("Sector Allocation")
                    sector_fig = go.Figure(data=[go.Pie(
                        labels=list(sector_weights.keys()),
                        values=list(sector_weights.values()),
                        textinfo='label+percent',
                        textposition='auto'
                    )])
                    sector_fig.update_layout(title="Portfolio Sector Allocation", height=400)
                    st.plotly_chart(sector_fig, use_container_width=True)
                
                # Factor Correlation Analysis
                st.header("Factor Correlation Analysis")
                factor_correlations = optimizer.calculate_factor_correlations(optimal_weights)
                
                if factor_correlations:
                    st.subheader("Portfolio Correlation to Factor ETFs")
                    
                    # Create correlation dataframe
                    corr_df = pd.DataFrame([
                        {'Factor': factor, 'Correlation': corr}
                        for factor, corr in factor_correlations.items()
                        if not pd.isna(corr)
                    ]).sort_values('Correlation', key=abs, ascending=False)
                    
                    if not corr_df.empty:
                        # Display correlation table
                        col1, col2 = st.columns([1, 1])
                        
                        with col1:
                            st.dataframe(corr_df.style.format({'Correlation': '{:.3f}'}), use_container_width=True)
                        
                        with col2:
                            # Create correlation bar chart
                            fig_corr = go.Figure()
                            
                            colors = ['green' if x > 0 else 'red' for x in corr_df['Correlation']]
                            
                            fig_corr.add_trace(go.Bar(
                                x=corr_df['Factor'],
                                y=corr_df['Correlation'],
                                marker_color=colors,
                                text=[f"{val:.3f}" for val in corr_df['Correlation']],
                                textposition='auto'
                            ))
                            
                            fig_corr.update_layout(
                                title="Factor Correlations",
                                xaxis_title="Factor ETF",
                                yaxis_title="Correlation",
                                height=400,
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_corr, use_container_width=True)
                        
                        # Factor insights
                        strongest_factor = corr_df.iloc[0]
                        st.info(f"Portfolio shows strongest correlation with **{strongest_factor['Factor']}** (r = {strongest_factor['Correlation']:.3f})")
                    
                else:
                    st.info("Factor correlation analysis requires sufficient overlapping data. Try a longer time period.")
                
                # Factor Attribution Analysis (ETF-based with PCA fallback)
                st.header("Advanced Factor Attribution Analysis")
                
                # Try Factor ETF attribution first
                factor_etf_results = optimizer.calculate_factor_etf_attribution(optimal_weights)
                
                if factor_etf_results:
                    st.subheader("Factor ETF Attribution (Industry Standard)")
                    
                    # Key metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model R²", f"{factor_etf_results['model_r_squared']:.3f}")
                        st.metric("Model Type", factor_etf_results['model_type'])
                    with col2:
                        st.metric("Alpha (Annual)", f"{factor_etf_results['alpha_annualized']*100:.2f}%")
                        st.metric("Systematic Risk", f"{factor_etf_results['systematic_risk_pct']:.1f}%")
                    with col3:
                        st.metric("Residual Vol", f"{factor_etf_results['residual_volatility']*100:.1f}%")
                        st.metric("Observations", factor_etf_results['observation_count'])
                    
                    # Factor exposures table and chart
                    exposures = factor_etf_results['factor_exposures']
                    exposure_df = pd.DataFrame([
                        {
                            'Factor': factor_name,
                            'Beta': f"{data['beta']:.3f}",
                            'Significance': data['significance'],
                            'Risk Contribution': f"{data['risk_contribution_pct']:.1f}%"
                        }
                        for factor_name, data in exposures.items()
                    ]).sort_values('Risk Contribution', key=lambda x: x.str.rstrip('%').astype(float), ascending=False)
                    
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.subheader("Factor Exposures (Betas)")
                        st.dataframe(exposure_df, use_container_width=True)
                    
                    with col2:
                        # Factor exposure bar chart
                        fig_exp = go.Figure()
                        betas = [exposures[factor]['beta'] for factor in exposures.keys()]
                        factor_names_list = list(exposures.keys())
                        colors = ['green' if x > 0 else 'red' for x in betas]
                        
                        fig_exp.add_trace(go.Bar(
                            x=factor_names_list,
                            y=betas,
                            marker_color=colors,
                            text=[f"{val:.3f}" for val in betas],
                            textposition='auto'
                        ))
                        fig_exp.update_layout(
                            title="Factor Exposures (Beta)",
                            xaxis_title="Factor",
                            yaxis_title="Beta (Exposure)",
                            height=400,
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig_exp, use_container_width=True)
                    
                    # Portfolio insights
                    if 'factor_insights' in factor_etf_results:
                        insights = factor_etf_results['factor_insights']
                        
                        st.subheader("Portfolio Style Analysis")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Portfolio Style", insights['portfolio_style'])
                        with col2:
                            st.metric("Risk Type", insights['risk_concentration'])
                        with col3:
                            st.metric("Dominant Factors", len(insights['dominant_factors']))
                        
                        # Dominant factors
                        if insights['dominant_factors']:
                            st.subheader("Key Factor Exposures")
                            for factor in insights['dominant_factors']:
                                direction_text = "Overweight" if factor['direction'] == "Overweight" else "Underweight"
                                st.write(f"**{factor['factor']}**: {direction_text} "
                                       f"(β = {factor['beta']:.3f}, {factor['strength']} significance)")
                        
                        # Actionable recommendations
                        if insights['actionable_recommendations']:
                            st.subheader("Portfolio Optimization Recommendations")
                            for rec in insights['actionable_recommendations']:
                                st.info(f"{rec}")
                    
                    # Show PCA as additional analysis
                    with st.expander("Additional PCA Analysis", expanded=False):
                        st.write("**Supplementary PCA analysis for comparison:**")
                        pca_results = optimizer.calculate_pca_factor_decomposition()
                        if pca_results:
                            # Mini PCA display
                            st.write(f"PCA explains {pca_results['total_variance_explained']*100:.1f}% of variance with {pca_results['n_components']} components")
                            if 'factor_interpretations' in pca_results:
                                for i, (factor_name, interp) in enumerate(list(pca_results['factor_interpretations'].items())[:2]):
                                    st.write(f"**{factor_name}**: {interp['financial_meaning']} ({interp['variance_explained']*100:.1f}%)")
                        else:
                            st.write("PCA analysis unavailable")
                
                else:
                    # Fallback to PCA if Factor ETF attribution fails
                    st.warning("Factor ETF attribution unavailable. Using PCA analysis as fallback.")
                    st.subheader("PCA Factor Decomposition Analysis")
                    pca_results = optimizer.calculate_pca_factor_decomposition()
                
                if not factor_etf_results and pca_results:
                    st.subheader("Principal Components Analysis")
                    
                    # Display variance explained
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.metric("Total Variance Explained", f"{pca_results['total_variance_explained']*100:.1f}%")
                        st.metric("Number of Components", pca_results['n_components'])
                        
                        # Variance explained table
                        variance_df = pd.DataFrame({
                            'Component': [f'PC{i+1}' for i in range(pca_results['n_components'])],
                            'Variance Explained': [f"{var*100:.1f}%" for var in pca_results['explained_variance_ratio']],
                            'Cumulative': [f"{cum*100:.1f}%" for cum in pca_results['cumulative_variance_explained']]
                        })
                        st.dataframe(variance_df, use_container_width=True)
                    
                    with col2:
                        # Scree plot - Variance explained by each component
                        fig_scree = go.Figure()
                        fig_scree.add_trace(go.Bar(
                            x=[f'PC{i+1}' for i in range(pca_results['n_components'])],
                            y=pca_results['explained_variance_ratio'] * 100,
                            name='Individual',
                            marker_color='lightblue'
                        ))
                        fig_scree.add_trace(go.Scatter(
                            x=[f'PC{i+1}' for i in range(pca_results['n_components'])],
                            y=pca_results['cumulative_variance_explained'] * 100,
                            mode='lines+markers',
                            name='Cumulative',
                            line=dict(color='red', width=2),
                            marker=dict(size=6)
                        ))
                        fig_scree.update_layout(
                            title="Variance Explained by Principal Components",
                            xaxis_title="Principal Component",
                            yaxis_title="Variance Explained (%)",
                            height=400
                        )
                        st.plotly_chart(fig_scree, use_container_width=True)
                    
                    # Factor loadings heatmap
                    st.subheader("Factor Loadings Matrix")
                    loadings_df = pd.DataFrame(
                        pca_results['components'].T,
                        index=pca_results['symbols'],
                        columns=[f'PC{i+1}' for i in range(pca_results['n_components'])]
                    )
                    
                    # Create heatmap
                    fig_loadings = go.Figure(data=go.Heatmap(
                        z=loadings_df.values,
                        x=loadings_df.columns,
                        y=loadings_df.index,
                        colorscale='RdBu',
                        zmid=0,
                        text=np.round(loadings_df.values, 3),
                        texttemplate='%{text}',
                        textfont={"size": 10},
                        colorbar=dict(title="Loading Strength")
                    ))
                    fig_loadings.update_layout(
                        title="Principal Component Loadings",
                        xaxis_title="Principal Components",
                        yaxis_title="Assets",
                        height=400
                    )
                    st.plotly_chart(fig_loadings, use_container_width=True)
                    
                    # Financial Interpretation of Factors
                    st.subheader("What Do These Factors Actually Mean?")
                    
                    if 'factor_interpretations' in pca_results and pca_results['factor_interpretations']:
                        interpretations = pca_results['factor_interpretations']
                        
                        # Show interpretations for the most important factors
                        for i, (factor_name, interp) in enumerate(list(interpretations.items())[:3]):
                            with st.expander(f"{factor_name}: {interp['financial_meaning']} ({interp['variance_explained']*100:.1f}% of risk)", expanded=i==0):
                                st.write(f"**Factor Type:** {interp['factor_type']}")
                                st.write(f"**What it means:** {interp['explanation']}")
                                
                                st.write("**Top Contributing Assets:**")
                                for driver in interp['top_drivers']:
                                    direction = "Positive" if driver['loading'] > 0 else "Negative"
                                    st.write(f"- **{driver['symbol']}** ({driver['name'][:30]}{'...' if len(driver['name']) > 30 else ''}): "
                                           f"{direction} influence ({driver['loading']:.3f})")
                                
                                # Add actionable insights
                                if interp['factor_type'] == 'Market Factor':
                                    st.info("**Investment Insight:** This factor represents systematic market risk that cannot be diversified away. Consider hedging strategies during high volatility periods.")
                                elif interp['factor_type'] == 'Sector/Style Factor':
                                    st.info("**Investment Insight:** This shows style/sector concentration. Consider rebalancing if you want to reduce this specific risk.")
                                elif interp['factor_type'] == 'Idiosyncratic Factor':
                                    st.info("**Investment Insight:** This represents company-specific risks that could be reduced through further diversification.")
                    
                    # Summary insights
                    st.subheader("Portfolio Risk Summary")
                    total_explained = pca_results['cumulative_variance_explained'][min(2, pca_results['n_components']-1)]
                    
                    risk_level = "Low" if total_explained < 0.85 else "Moderate" if total_explained < 0.95 else "High"
                    diversification = "Well-diversified" if total_explained < 0.85 else "Moderately concentrated" if total_explained < 0.95 else "Concentrated"
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Risk Concentration", risk_level)
                    with col2:
                        st.metric("Diversification Level", diversification)
                    with col3:
                        st.metric("Systematic Risk", f"{total_explained*100:.1f}%")
                    
                    if total_explained > 0.95:
                        st.warning("**High Concentration Risk:** Your portfolio is dominated by a few factors. Consider diversifying across more asset classes or sectors.")
                    elif total_explained < 0.85:
                        st.success("**Good Diversification:** Your portfolio risk is spread across multiple factors, reducing concentration risk.")
                    else:
                        st.info("**Moderate Concentration:** Your portfolio has reasonable diversification but could benefit from some optimization.")
                
                else:
                    st.info("PCA factor decomposition requires sufficient return data.")
                
                # Implied Volatility Analysis (if data available)
                if iv_data:
                    st.header("Options Implied Volatility Analysis")
                    
                    st.subheader("30-Day Implied Volatility")
                    
                    # Create IV dataframe
                    iv_df = pd.DataFrame([
                        {
                            'Symbol': data['symbol'],
                            'Current IV (%)': f"{data['avg_iv']:.2f}",
                            'Historical Vol (%)': f"{data.get('historical_vol', 0):.2f}",
                            'IV vs HV': f"{data.get('iv_vs_hv', 0):+.2f}%",
                            'Volume': f"{data.get('volume', 0):.2f}%",
                            'Open Interest': f"{data.get('open_interest', 0):.2f}%",
                            'Date': data.get('date', 'N/A')
                        }
                        for data in iv_data.values()
                    ])
                    
                    if not iv_df.empty:
                        st.dataframe(iv_df, use_container_width=True)
                        
                        # Create IV visualization
                        fig_iv = go.Figure()
                        
                        symbols_list = list(iv_data.keys())
                        avg_iv_values = [iv_data[symbol]['avg_iv'] for symbol in symbols_list]
                        
                        fig_iv.add_trace(go.Bar(
                            x=symbols_list,
                            y=avg_iv_values,
                            name='Average IV',
                            marker_color='#007bff',
                            text=[f"{val:.1f}%" for val in avg_iv_values],
                            textposition='auto'
                        ))
                        
                        fig_iv.update_layout(
                            title="Implied Volatility Comparison (30-Day)",
                            xaxis_title="Symbol",
                            yaxis_title="Implied Volatility (%)",
                            height=400,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig_iv, use_container_width=True)
                    else:
                        st.info("No IV data available for equity symbols in portfolio")
                else:
                    st.header("Options Implied Volatility Analysis")
                    st.info("IV data will be automatically fetched during optimization for equity symbols in your portfolio.")
                    
                    # Manual refresh option
                    if st.button("Refresh IV Data"):
                        fresh_iv_data = optimizer.fetch_iv_data(tenor_days=30, show_progress=True)
                        if fresh_iv_data:
                            st.success("IV data refreshed successfully!")
                            st.rerun()  # Refresh the page to show new data
                
                # Historical data preview
                st.header("Historical Data Preview")
                st.dataframe(data.tail(10), use_container_width=True)
                
                # Export options
                st.header("Export Options")
                
                export_col1, export_col2 = st.columns([1, 1])
                
                with export_col1:
                    if st.button("Download Excel Report"):
                        # Create Excel export
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                            weights_df.to_excel(writer, sheet_name='Portfolio Weights', index=False)
                            
                            metrics_df = pd.DataFrame({
                                'Metric': ['Annual Return', 'Annual Volatility', 'Sharpe Ratio', 'Max Drawdown', 'Calmar Ratio', 'Sortino Ratio'],
                                'Value': [f"{metrics['annual_return']*100:.2f}%", f"{metrics['annual_volatility']*100:.2f}%", 
                                         f"{metrics['sharpe_ratio']:.3f}", f"{metrics['max_drawdown']*100:.2f}%",
                                         f"{metrics['calmar_ratio']:.3f}", f"{metrics['sortino_ratio']:.3f}"]
                            })
                            metrics_df.to_excel(writer, sheet_name='Performance Metrics', index=False)
                            
                            data.to_excel(writer, sheet_name='Historical Data')
                        
                        st.download_button(
                            label="Download Excel",
                            data=output.getvalue(),
                            file_name=f"portfolio_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                
                with export_col2:
                    # CSV export
                    csv = weights_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"portfolio_weights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")
                st.info("Please check your symbols and try again.")
    
    else:
        st.success("Enter your ETF/stock symbols in the sidebar and click 'Optimize Portfolio' to get started!")
        
        # Sample portfolio demonstration
        st.header("About This Tool")
        st.info("This tool uses Modern Portfolio Theory to optimize your portfolio allocation based on historical data and your chosen objective.")
        
        st.markdown("""
        **Features:**
        - Multiple optimization objectives (Sharpe ratio, minimum variance, drawdown minimization)
        - **Black-Litterman Model**: Incorporates your market views with uncertainty
        - Comprehensive performance metrics and factsheet
        - **Auto-fetch symbol information** (market cap, P/E, beta, sector, etc.)
        - **Auto-fetch implied volatility data** for individual stocks
        - Interactive charts and visualizations
        - Export capabilities (Excel, CSV)
        - Real-time data from Yahoo Finance
        
        **How to use:**
        1. Enter your symbols in the sidebar (mix stocks and ETFs)
        2. Choose your optimization objective
        3. **For Black-Litterman**: Express your market views and confidence levels
        4. Select the data period for analysis
        5. Click "Optimize Portfolio"
        6. Review results, factsheet, and IV analysis
        7. Export if needed
        
        **Black-Litterman Model:**
        - Combines market equilibrium with your personal views
        - Accounts for uncertainty in your predictions
        - More intuitive than traditional mean-variance optimization
        - Uses market cap weights as starting point (when available)
        
        **Pro tip:** Include individual stocks (AAPL, GOOGL, etc.) for IV analysis!
        """)

if __name__ == "__main__":
    main()