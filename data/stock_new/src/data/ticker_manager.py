import yfinance as yf
import pandas as pd
import polars as pl
import json
import sqlite3
import logging
from typing import Dict, Optional, List
from datetime import datetime, timedelta, date
import os
import duckdb
import requests
from bs4 import BeautifulSoup
import time
import numpy as np
from pathlib import Path
from .download_nasdaq import download_nasdaq_screener

class TickerManager:
    def __init__(self, db=None):
        self.logger = logging.getLogger(__name__)
        self.db = db
        
        # Set up data directories
        self.base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.data_dir = self.base_dir / 'data'
        self.exports_dir = self.data_dir / 'exports'
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.exports_dir.mkdir(exist_ok=True)
        
        # Initialize data
        self.tickers = self.load_tickers()
        self.categories = self.initialize_categories()
        self.cache_file = self.exports_dir / "tickers_cache.json"
        
        logging.info(f"TickerManager initialized with {self._count_total_tickers()} tickers")

    def load_tickers(self) -> List[str]:
        """Load tickers from NASDAQ data."""
        self.logger.info("Loading tickers from cache")
        try:
            # Look for existing NASDAQ screener files
            nasdaq_files = list(self.data_dir.glob('nasdaq_screener_*.csv'))
            
            if not nasdaq_files:
                self.logger.info("No NASDAQ screener file found. Downloading...")
                filename = download_nasdaq_screener()
                if not filename:
                    raise FileNotFoundError("Failed to download NASDAQ screener data")
                nasdaq_files = [filename]
                self.logger.info(f"Downloaded new NASDAQ data to {filename}")
            
            # Use the most recent file
            latest_file = max(nasdaq_files, key=lambda x: int(x.stem.split('_')[-1]))
            self.logger.info(f"Loading NASDAQ data from {latest_file}")
            
            # Read the CSV file
            df = pd.read_csv(latest_file)
            self.logger.info(f"Read CSV file with {len(df)} rows and columns: {', '.join(df.columns)}")
            
            # Extract unique ticker symbols
            if 'symbol' in df.columns:
                tickers = df['symbol'].unique().tolist()
                self.logger.info(f"Using 'symbol' column for tickers")
            elif 'Symbol' in df.columns:
                tickers = df['Symbol'].unique().tolist()
                self.logger.info(f"Using 'Symbol' column for tickers")
            else:
                available_columns = ', '.join(df.columns)
                raise ValueError(f"No symbol column found in NASDAQ data. Available columns: {available_columns}")
            
            self.logger.info(f"Loaded {len(tickers)} unique tickers")
            return tickers
            
        except Exception as e:
            self.logger.error(f"Error loading tickers: {e}")
            return []

    def initialize_categories(self) -> Dict[str, List[str]]:
        """Initialize ticker categories."""
        try:
            # Create basic categories (can be expanded later)
            categories = {
                'All': self.tickers,
                'Technology': ['AAPL', 'MSFT', 'GOOG', 'GOOGL', 'META', 'NVDA', 'INTC', 'AMD', 'TSM', 'AVGO'],
                'Finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'AXP', 'V', 'MA'],
                'Healthcare': ['JNJ', 'PFE', 'MRK', 'ABBV', 'LLY', 'TMO', 'DHR', 'BMY', 'UNH', 'ABT'],
                'Consumer': ['AMZN', 'WMT', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'MCD', 'SBUX', 'HD'],
                'Industrial': ['GE', 'BA', 'CAT', 'HON', 'MMM', 'UPS', 'FDX', 'LMT', 'RTX', 'DE']
            }
            
            # Filter out any tickers that don't exist in our main list
            for category in categories:
                if category != 'All':
                    categories[category] = [
                        ticker for ticker in categories[category]
                        if ticker in self.tickers
                    ]
            
            self.logger.info(f"Initialized {len(categories)} categories")
            return categories
            
        except Exception as e:
            self.logger.error(f"Error initializing categories: {e}")
            return {'All': self.tickers}

    def _count_total_tickers(self) -> int:
        """Count total number of unique tickers across all categories."""
        all_tickers = set()
        for tickers in self.categories.values():
            all_tickers.update(tickers)
        return len(all_tickers)

    def get_category_names(self) -> List[str]:
        """Get list of category names."""
        return list(self.categories.keys())

    def get_tickers_in_category(self, category: str) -> List[str]:
        """Get tickers in a specific category."""
        return self.categories.get(category, [])

    def validate_ticker(self, ticker: str) -> bool:
        """Validate if a ticker exists and is active."""
        try:
            # First check if ticker is in our list
            if ticker not in self.tickers:
                self.logger.warning(f"Ticker {ticker} not found in NASDAQ data")
                return False
                
            # Then verify with yfinance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if info and 'regularMarketPrice' in info:
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Error validating ticker {ticker}: {e}")
            return False

    def get_quotes_df(self, tickers: List[str], start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
        """Get historical quotes for multiple tickers."""
        try:
            all_data = []
            total = len(tickers)
            
            for i, ticker in enumerate(tickers, 1):
                self.logger.info(f"Fetching data for {ticker} ({i}/{total})")
                
                try:
                    # Validate ticker first
                    if not self.validate_ticker(ticker):
                        self.logger.warning(f"Skipping invalid ticker: {ticker}")
                        continue
                    
                    # Get data with retry mechanism
                    for attempt in range(3):
                        try:
                            stock = yf.Ticker(ticker)
                            df = stock.history(start=start_date, end=end_date, interval=interval)
                            
                            if not df.empty:
                                df['Symbol'] = ticker
                                all_data.append(df)
                                break
                                
                            time.sleep(1)  # Rate limiting
                            
                        except Exception as e:
                            self.logger.error(f"Attempt {attempt + 1} failed for {ticker}: {e}")
                            if attempt < 2:
                                time.sleep(2 ** attempt)  # Exponential backoff
                            continue
                    
                except Exception as e:
                    self.logger.error(f"Error processing ticker {ticker}: {e}")
                    continue
                
            if all_data:
                result = pd.concat(all_data)
                result = result.reset_index()
                return result
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error getting quotes: {e}")
            return pd.DataFrame()

    def get_single_ticker_data(self, ticker: str, start_date: str, end_date: str, interval: str = '1d') -> Optional[Dict]:
        """Get data for a single ticker with additional information."""
        try:
            # Validate ticker first
            if not self.validate_ticker(ticker):
                self.logger.warning(f"Invalid ticker: {ticker}")
                return None
            
            # Get historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date, interval=interval)
            
            if hist.empty:
                self.logger.warning(f"No historical data found for {ticker}")
                return None
            
            # Get current info
            info = stock.info
            
            # Prepare response
            response = {
                'symbol': ticker,
                'start_price': hist['Close'].iloc[0],
                'last_price': hist['Close'].iloc[-1],
                'historical_data': hist.reset_index().to_dict('records'),
                'company_name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'website': info.get('website', ''),
                'description': info.get('longBusinessSummary', '')
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting data for {ticker}: {e}")
            return None

    def get_simple_quote(self, symbol: str) -> Optional[Dict]:
        """Get a simple quote using yfinance."""
        try:
            logging.info(f"Fetching quote for {symbol}")
            
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            
            if not info:
                logging.error(f"No info returned for {symbol}")
                return None
                
            return {
                'symbol': symbol,
                'price': info.last_price,
                'change': info.last_price - info.previous_close,
                'change_percent': ((info.last_price - info.previous_close) / info.previous_close) * 100,
                'volume': info.last_volume,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            logging.error(f"Error fetching quote for {symbol}: {e}")
            return None

    def get_available_symbols(self) -> list:
        """Get a list of test symbols."""
        return ['AAPL', 'MSFT', 'GOOGL']

    def get_historical_data(self, ticker: str, start_date: datetime, end_date: datetime, interval: str = '1d') -> pd.DataFrame:
        """
        Get historical data for a ticker with improved error handling and validation.
        """
        logging.info(f"Fetching historical data for {ticker}")
        
        # Convert dates to datetime if they're strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Set fixed current date since system is in 2025
        current_date = datetime(2024, 3, 25, 23, 59, 59)
        
        # Adjust end date if it's after current date
        if end_date > current_date:
            logging.warning(f"Adjusting end date from {end_date} to current date {current_date}")
            end_date = current_date
            
        # Ensure we have at least a 7-day range to avoid yfinance API issues
        min_days = 7
        if (end_date - start_date).days < min_days:
            start_date = end_date - timedelta(days=min_days)
            logging.info(f"Adjusting start date to ensure minimum {min_days}-day range: {start_date}")
            
        # Ensure start date is not in the future
        if start_date > current_date:
            start_date = current_date - timedelta(days=30)
            logging.warning(f"Start date was in future, adjusted to {start_date}")
            
        # Ensure start date is before end date
        if start_date > end_date:
            start_date = end_date - timedelta(days=30)
            logging.warning(f"Start date was after end date, adjusted to {start_date}")
            
        # Limit maximum date range to 2 years
        max_days = 365 * 2
        if (end_date - start_date).days > max_days:
            start_date = end_date - timedelta(days=max_days)
            logging.warning(f"Date range too large, adjusted start date to {start_date}")
            
        logging.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
        
        # Create empty DataFrame with correct structure
        empty_df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                                       'Symbol', 'Daily_Return', 'Volatility',
                                       'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26',
                                       'MACD', 'MACD_Signal', 'RSI', 'BB_Middle', 'BB_Upper',
                                       'BB_Lower', 'Volume_MA', 'Volume_Ratio'])

        # Create a session with custom headers and timeouts
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        session.timeout = (5, 10)  # (connect timeout, read timeout)

        # Initialize retry mechanism
        max_attempts = 3
        attempt = 0
        backoff_factor = 2
        initial_wait = 3

        while attempt < max_attempts:
            attempt += 1
            try:
                # First validate if the ticker exists using Yahoo Finance quote endpoint
                validation_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=1d&interval=1d"
                validation_response = session.get(validation_url, timeout=5)
                
                if validation_response.status_code != 200:
                    logging.error(f"Ticker {ticker} appears to be invalid (Status code: {validation_response.status_code})")
                    return empty_df
                
                validation_data = validation_response.json()
                if 'chart' not in validation_data or 'error' in validation_data:
                    logging.error(f"Ticker {ticker} validation failed: {validation_data.get('error', 'Unknown error')}")
                    return empty_df

                # Convert dates to Unix timestamp for API
                period1 = int(start_date.timestamp())
                period2 = int(end_date.timestamp())
                
                # Construct Yahoo Finance API URL with all necessary parameters
                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
                params = {
                    'period1': period1,
                    'period2': period2,
                    'interval': interval,
                    'events': 'history',
                    'includeAdjustedClose': True
                }
                
                response = session.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'chart' in data and 'result' in data['chart'] and data['chart']['result']:
                        result = data['chart']['result'][0]
                        
                        # Extract timestamp and price data
                        timestamps = pd.to_datetime(result['timestamp'], unit='s')
                        quotes = result['indicators']['quote'][0]
                        
                        # Create DataFrame with all available data
                        df = pd.DataFrame({
                            'Date': timestamps,
                            'Open': quotes.get('open', []),
                            'High': quotes.get('high', []),
                            'Low': quotes.get('low', []),
                            'Close': quotes.get('close', []),
                            'Volume': quotes.get('volume', []),
                            'Symbol': ticker
                        })
                        
                        # Remove any rows with NaN values
                        df = df.dropna()
                        
                        if not df.empty:
                            # Calculate additional metrics
                            df['Daily_Return'] = df['Close'].pct_change()
                            df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
                            
                            # Add technical indicators
                            df = self.add_technical_indicators(df)
                            
                            # Convert Date to string format
                            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
                            
                            return df
                    
                    logging.warning(f"No valid data in response for {ticker}")
                    
                elif response.status_code == 429:
                    logging.warning(f"Rate limit hit for {ticker}, waiting before retry...")
                    time.sleep(initial_wait * (backoff_factor ** (attempt - 1)))
                    continue
                    
                else:
                    logging.error(f"Failed to fetch data for {ticker}: HTTP {response.status_code}")
                    
            except requests.exceptions.Timeout:
                logging.warning(f"Timeout while fetching {ticker}, attempt {attempt}")
                if attempt < max_attempts:
                    wait_time = initial_wait * (backoff_factor ** (attempt - 1))
                    time.sleep(wait_time)
                continue
                
            except Exception as e:
                logging.error(f"Error fetching data for {ticker}: {str(e)}")
                if attempt < max_attempts:
                    wait_time = initial_wait * (backoff_factor ** (attempt - 1))
                    time.sleep(wait_time)
                continue
        
        logging.error(f"Failed to fetch data for {ticker} after {max_attempts} attempts")
        return empty_df

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        try:
            # Moving averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential moving averages
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            
            # Relative Strength Index (RSI)
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            df['BB_Upper'] = df['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
            df['BB_Lower'] = df['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
            
            # Volume indicators
            df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
            
            # Price changes
            df['Daily_Return'] = df['Close'].pct_change()
            df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
            
            return df

        except Exception as e:
            logging.error(f"Error calculating technical indicators: {e}")
            return df

    def export_to_csv(self, df: pd.DataFrame, filename: str) -> str:
        """Export data to CSV."""
        try:
            path = self.exports_dir / f"{filename}.csv"
            df.to_csv(path, index=False)
            return path
        except Exception as e:
            logging.error(f"Error exporting to CSV: {e}")
            raise

    def export_to_json(self, df: pd.DataFrame, filename: str) -> str:
        """Export data to JSON."""
        try:
            path = self.exports_dir / f"{filename}.json"
            df.to_json(path, orient='records', indent=2)
            return path
        except Exception as e:
            logging.error(f"Error exporting to JSON: {e}")
            raise

    def export_to_sqlite(self, df: pd.DataFrame, filename: str, table_name: str) -> str:
        """Export data to SQLite."""
        try:
            path = self.exports_dir / f"{filename}.db"
            conn = sqlite3.connect(path)
            df.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            return path
        except Exception as e:
            logging.error(f"Error exporting to SQLite: {e}")
            raise

    def convert_to_polars(self, df: pd.DataFrame) -> pl.DataFrame:
        """Convert pandas DataFrame to polars DataFrame."""
        try:
            return pl.from_pandas(df)
        except Exception as e:
            logging.error(f"Error converting to Polars: {e}")
            raise

    def export_to_duckdb(self, df: pd.DataFrame, filename: str, table_name: str) -> str:
        """Export data to DuckDB."""
        try:
            path = self.exports_dir / f"{filename}.duckdb"
            conn = duckdb.connect(path)
            conn.register('temp_table', df)
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_table")
            conn.close()
            return path
        except Exception as e:
            logging.error(f"Error exporting to DuckDB: {e}")
            raise

# Add this line at the end of the file
__all__ = ['TickerManager'] 