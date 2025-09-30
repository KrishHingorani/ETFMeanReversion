
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import random
from bs4 import BeautifulSoup
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

class AlphaQueryIVScraper:
    def __init__(self):
        """
        Initialize AlphaQuery IV scraper with session rotation to avoid limits
        """
        self.base_url = "https://www.alphaquery.com"
        self.session = None
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        self._create_new_session()
    
    def _create_new_session(self):
        """
        Create a new session with random headers to simulate incognito Browse
        """
        self.session = requests.Session()
        
        # Random user agent
        user_agent = random.choice(self.user_agents)
        
        headers = {
            'User-Agent': user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache'
        }
        
        self.session.headers.update(headers)
        
        # Add random delay
        time.sleep(random.uniform(1, 5))
    
    def get_iv_data(self, symbol, tenor_days=30, refresh_session=True):
        """
        Scrape IV data from AlphaQuery for a specific symbol and tenor
        
        Parameters:
        symbol (str): Stock symbol (e.g., 'AAPL', 'SPY')
        tenor_days (int): Days for IV calculation (max 180)
        refresh_session (bool): Whether to create new session (simulates incognito)
        
        Returns:
        dict: Dictionary containing IV data and metrics
        """
        if refresh_session:
            self._create_new_session()
        
        if tenor_days > 180:
            print(f"Warning: Maximum tenor is 180 days, using 180 instead of {tenor_days}")
            tenor_days = 180
        
        try:
            # Construct URL
            url = f"{self.base_url}/stock/{symbol.upper()}/volatility-option-statistics/{tenor_days}-day/iv-mean"
            
            print(f"Scraping IV data for {symbol} ({tenor_days}-day tenor)")
            print(f"URL: {url}")
            
            # Make request
            response = self.session.get(url, timeout=30)
            
            if response.status_code != 200:
                print(f"Failed to fetch data. Status code: {response.status_code}")
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract current IV value from the main display
            iv_data = self._extract_current_iv(soup, symbol, tenor_days)
            
            # Extract historical data via AJAX call
            historical_data = self._get_historical_iv_data(symbol, tenor_days)
            
            if historical_data:
                iv_data['historical_data'] = historical_data
            
            # Extract additional metrics from the page
            metrics = self._extract_volatility_metrics(soup)
            if metrics:
                iv_data.update(metrics)
            
            return iv_data
            
        except Exception as e:
            print(f"Error scraping IV data for {symbol}: {e}")
            return None
    
    def _extract_current_iv(self, soup, symbol, tenor_days):
        """
        Extract current IV value from the page
        """
        iv_data = {
            'symbol': symbol,
            'tenor_days': tenor_days,
            'timestamp': datetime.now().isoformat(),
            'current_iv': None,
            'date': None
        }
        
        try:
            # Check for subscription/access modal
            if "free access has expired" in soup.get_text().lower():
                print("⚠️  Free access expired - may have hit limits")
            
            # Look for the main IV value in the page text
            page_text = soup.get_text()
            
            # Search for patterns like "Implied Volatility (Mean)" followed by a number
            import re
            
            # Pattern for IV value near "Implied Volatility" text
            iv_patterns = [
                r'Implied Volatility.*?(\d+\.\d+)',
                r'IV.*?Mean.*?(\d+\.\d+)',
                r'(\d+\.\d+).*?Implied Volatility',
                r'volatility.*?(\d+\.\d+)',
                r'(\d\.\d{4})'  # Pattern like 0.1533
            ]
            
            for pattern in iv_patterns:
                matches = re.findall(pattern, page_text, re.IGNORECASE)
                for match in matches:
                    try:
                        value = float(match)
                        if 0 <= value <= 5:  # Reasonable IV range
                            iv_data['current_iv'] = value
                            print(f"Found IV value: {value}")
                            break
                    except:
                        continue
                if iv_data['current_iv']:
                    break
            
            # Also try to find values in HTML structure
            if not iv_data['current_iv']:
                # Look for specific elements that might contain the IV value
                for element in soup.find_all(['span', 'div', 'td']):
                    text = element.get_text().strip()
                    if text and re.match(r'^\d+\.\d{4}$', text):  # Pattern like 0.1533
                        try:
                            value = float(text)
                            if 0 <= value <= 2:  # Reasonable IV range
                                iv_data['current_iv'] = value
                                print(f"Found IV value in element: {value}")
                                break
                        except:
                            continue
            
            # Extract date - look for 2025 dates
            date_matches = re.findall(r'(\d{4}-\d{2}-\d{2})', page_text)
            if date_matches:
                iv_data['date'] = date_matches[0]
                    
        except Exception as e:
            print(f"Error extracting current IV: {e}")
        
        return iv_data
    
    def _get_historical_iv_data(self, symbol, tenor_days):
        """
        Try to get historical IV data via AJAX endpoint
        """
        try:
            # Try the AJAX endpoint for chart data
            ajax_url = f"{self.base_url}/data/option-statistic-chart"
            
            params = {
                'symbol': symbol.upper(),
                'metric': 'iv-mean',
                'period': f'{tenor_days}-day'
            }
            
            response = self.session.get(ajax_url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                print(f"AJAX request failed with status {response.status_code}")
                
        except Exception as e:
            print(f"Error fetching historical data: {e}")
        
        return None
    
    def _extract_volatility_metrics(self, soup):
        """
        Extract additional volatility metrics from the page
        """
        metrics = {}
        
        try:
            # Look for metrics table or similar structures
            tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    if len(cells) >= 2:
                        label = cells[0].get_text().strip().lower()
                        value_text = cells[1].get_text().strip()
                        
                        # Try to extract numeric values
                        try:
                            value = float(value_text)
                            
                            if 'historical volatility' in label:
                                metrics['historical_volatility'] = value
                            elif 'calls' in label and 'iv' in label:
                                metrics['iv_calls'] = value
                            elif 'puts' in label and 'iv' in label:
                                metrics['iv_puts'] = value
                            elif 'volume' in label:
                                metrics['volume'] = value
                            elif 'open interest' in label:
                                metrics['open_interest'] = value
                                
                        except:
                            continue
            
        except Exception as e:
            print(f"Error extracting metrics: {e}")
        
        return metrics
    
    def get_multiple_symbols_iv(self, symbols, tenor_days=30, delay_range=(2, 5)):
        """
        Get IV data for multiple symbols with session rotation
        
        Parameters:
        symbols (list): List of stock symbols
        tenor_days (int): Days for IV calculation
        delay_range (tuple): Min and max delay between requests
        
        Returns:
        dict: Dictionary with symbols as keys and IV data as values
        """
        results = {}
        
        for i, symbol in enumerate(symbols):
            print(f"\nProcessing {i+1}/{len(symbols)}: {symbol}")
            
            # Create new session every few requests
            refresh = (i % 3 == 0)
            
            data = self.get_iv_data(symbol, tenor_days, refresh_session=refresh)
            if data:
                results[symbol] = data
                print(f"✓ Successfully scraped {symbol}: IV = {data.get('current_iv', 'N/A')}")
            else:
                print(f"✗ Failed to scrape {symbol}")
            
            # Random delay between requests
            if i < len(symbols) - 1:  # Don't delay after last request
                delay = random.uniform(delay_range[0], delay_range[1])
                print(f"Waiting {delay:.1f}s before next request...")
                time.sleep(delay)
        
        return results
    
    def get_iv_term_structure(self, symbol, tenors=[10, 14, 20, 30, 60, 90, 120, 180]):
        """
        Get IV data for different tenors to build term structure
        
        Parameters:
        symbol (str): Stock symbol
        tenors (list): List of tenor days
        
        Returns:
        pandas.DataFrame: Term structure data
        """
        print(f"\nBuilding IV term structure for {symbol}")
        
        term_structure_data = []
        
        for i, tenor in enumerate(tenors):
            print(f"Fetching {tenor}-day IV...")
            
            # Refresh session every few requests
            refresh = (i % 2 == 0)
            
            data = self.get_iv_data(symbol, tenor, refresh_session=refresh)
            
            if data and data.get('current_iv'):
                term_structure_data.append({
                    'tenor_days': tenor,
                    'iv': data['current_iv'],
                    'date': data.get('date'),
                    'timestamp': data['timestamp']
                })
                print(f"✓ {tenor}-day IV: {data['current_iv']}")
            else:
                print(f"✗ Failed to get {tenor}-day IV")
            
            # Delay between requests
            if i < len(tenors) - 1:
                delay = random.uniform(3, 6)
                time.sleep(delay)
        
        if term_structure_data:
            df = pd.DataFrame(term_structure_data)
            return df
        else:
            print("No term structure data collected")
            return None
    
    def export_data(self, data, filename, format='csv'):
        """
        Export scraped data to file
        
        Parameters:
        data (dict or pandas.DataFrame): Data to export
        filename (str): Output filename
        format (str): Export format ('csv', 'excel', 'json')
        """
        try:
            if isinstance(data, pd.DataFrame):
                if format == 'excel':
                    data.to_excel(f"{filename}.xlsx", index=False)
                elif format == 'json':
                    data.to_json(f"{filename}.json", orient='records', indent=2)
                else:
                    data.to_csv(f"{filename}.csv", index=False)
            else:
                if format == 'json':
                    with open(f"{filename}.json", 'w') as f:
                        json.dump(data, f, indent=2, default=str)
                elif format == 'excel':
                    df = pd.DataFrame([data])
                    df.to_excel(f"{filename}.xlsx", index=False)
                else:
                    df = pd.DataFrame([data])
                    df.to_csv(f"{filename}.csv", index=False)
            
            print(f"Data exported to {filename}.{format}")
            
        except Exception as e:
            print(f"Error exporting data: {e}")

    def get_term_structures_for_symbols(self, symbols, tenors=[10, 20, 30, 60, 90]):
        """
        Get IV term structures for multiple symbols
        
        Parameters:
        symbols (list): List of stock symbols
        tenors (list): List of tenor days
        
        Returns:
        dict: Dictionary with symbols as keys and DataFrames as values
        """
        print(f"\n🚀 Getting IV Term Structures for {len(symbols)} symbols")
        print(f"Tenors: {tenors}")
        print("=" * 60)
        
        all_term_structures = {}
        
        for i, symbol in enumerate(symbols):
            print(f"\n[{i+1}/{len(symbols)}] Processing {symbol}...")
            
            # Get term structure for this symbol
            term_structure = self.get_iv_term_structure(symbol, tenors)
            
            if term_structure is not None and not term_structure.empty:
                all_term_structures[symbol] = term_structure
                print(f"✅ {symbol}: Got {len(term_structure)} data points")
                
                # Display preview
                if not term_structure.empty:
                    print(f"   Preview: {term_structure['tenor_days'].min()}d-{term_structure['tenor_days'].max()}d")
                    print(f"   IV range: {term_structure['iv'].min():.4f} - {term_structure['iv'].max():.4f}")
            else:
                print(f"❌ {symbol}: No term structure data")
            
            # Delay between symbols
            if i < len(symbols) - 1:
                delay = random.uniform(3, 6)
                print(f"   ⏳ Waiting {delay:.1f}s...")
                time.sleep(delay)
        
        return all_term_structures
    
    def create_combined_term_structure_table(self, term_structures):
        """
        Create a combined DataFrame from multiple term structures
        
        Parameters:
        term_structures (dict): Dict of symbol -> DataFrame
        
        Returns:
        pandas.DataFrame: Combined term structure table
        """
        if not term_structures:
            return None
        
        combined_data = []
        
        for symbol, df in term_structures.items():
            if df is not None and not df.empty:
                # Add symbol column to each row
                symbol_df = df.copy()
                symbol_df['symbol'] = symbol
                combined_data.append(symbol_df)
        
        if combined_data:
            combined_df = pd.concat(combined_data, ignore_index=True)
            
            # Reorder columns
            cols = ['symbol', 'tenor_days', 'iv', 'date', 'timestamp']
            combined_df = combined_df[cols]
            
            # Sort by symbol, then tenor
            combined_df = combined_df.sort_values(['symbol', 'tenor_days'])
            
            return combined_df
        
        return None
    
    def create_pivot_term_structure(self, combined_df):
        """
        Create a pivot table showing IV by symbol and tenor
        
        Parameters:
        combined_df (DataFrame): Combined term structure data
        
        Returns:
        pandas.DataFrame: Pivot table with symbols as rows, tenors as columns
        """
        if combined_df is None or combined_df.empty:
            return None
        
        # Create pivot table
        pivot_df = combined_df.pivot_table(
            values='iv', 
            index='symbol', 
            columns='tenor_days', 
            aggfunc='first'
        )
        
        return pivot_df

def plot_term_structures(combined_df, title="Implied Volatility Term Structure"):
    """
    Plots the IV term structure for multiple symbols using Plotly.

    Parameters:
    combined_df (pd.DataFrame): DataFrame containing symbol, tenor_days, and iv.
    title (str): The title of the plot.
    """
    if combined_df is None or combined_df.empty:
        print("Cannot plot, DataFrame is empty.")
        return

    print(f"\n🎨 Generating interactive plot: '{title}'")

    fig = px.line(
        combined_df,
        x='tenor_days',
        y='iv',
        color='symbol',
        markers=True,  # Add markers to data points
        labels={
            'tenor_days': 'Tenor (Days)',
            'iv': 'Implied Volatility',
            'symbol': 'Symbol'
        },
        title=title
    )

    # Enhance layout
    fig.update_layout(
        xaxis_title='Tenor (Days)',
        yaxis_title='Implied Volatility',
        yaxis_tickformat='.2%',  # Format y-axis as percentage (e.g., 0.15 -> 15.00%)
        legend_title_text='Symbols',
        hovermode='x unified', # Improved hover experience
        template='plotly_white'
    )
    
    # Show the figure
    fig.show()

def main():
    """
    Enhanced usage with prettier output and term structures for multiple symbols
    """
    scraper = AlphaQueryIVScraper()
    
    print("🚀 AlphaQuery IV Scraper - Enhanced Edition")
    print("=" * 50)
    
    # Test symbols for term structure analysis
    symbols = ['SPY', 'QQQ', 'TSLA', 'AAPL', 'MSFT']
    tenors = [10, 20, 30, 60, 90]
    
    print(f"\n📊 Getting term structures for: {', '.join(symbols)}")
    print(f"📈 Tenors: {tenors}")
    
    # Get term structures for all symbols
    term_structures = scraper.get_term_structures_for_symbols(symbols, tenors)
    
    if term_structures:
        print(f"\n✅ Successfully got term structures for {len(term_structures)} symbols")
        
        # Create combined DataFrame
        combined_df = scraper.create_combined_term_structure_table(term_structures)
        
        if combined_df is not None:
            print(f"\n📋 Combined Term Structure Data ({len(combined_df)} rows):")
            print("-" * 70)
            print(combined_df.to_string(index=False, float_format='%.4f'))
            
            # Create pivot table
            pivot_df = scraper.create_pivot_term_structure(combined_df)
            
            if pivot_df is not None:
                print(f"\n📊 IV Term Structure Pivot Table:")
                print("-" * 50)
                print(pivot_df.to_string(float_format='%.4f'))
                
                # --- ADDED VISUALIZATION ---
                plot_term_structures(combined_df, title=f"IV Term Structure Comparison ({datetime.now().strftime('%Y-%m-%d')})")
                
                # Export both formats
                scraper.export_data(combined_df, "combined_term_structures", "csv")
                scraper.export_data(pivot_df, "iv_term_structure_pivot", "csv")
                
                print(f"\n💾 Exported files:")
                print("   - combined_term_structures.csv (detailed data)")
                print("   - iv_term_structure_pivot.csv (pivot table)")
        
        # Individual term structure analysis
        print(f"\n📈 Individual Term Structure Analysis:")
        print("=" * 50)
        
        for symbol, df in term_structures.items():
            if df is not None and not df.empty:
                print(f"\n{symbol}:")
                print(f"  Data points: {len(df)}")
                print(f"  IV range: {df['iv'].min():.4f} - {df['iv'].max():.4f}")
                print(f"  Tenor range: {df['tenor_days'].min()}d - {df['tenor_days'].max()}d")
                
                # Show actual data
                print("  Term Structure:")
                for _, row in df.iterrows():
                    print(f"    {int(row['tenor_days']):3d}d: {row['iv']:.4f}")
    
    else:
        print("❌ No term structure data obtained")
    
    print(f"\n🎯 Analysis Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
