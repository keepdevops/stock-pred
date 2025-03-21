import yfinance as yf
import pandas as pd
import polars as pl
import json
import sqlite3
import logging
from typing import Dict, Optional, List
from datetime import datetime
import os
import duckdb
import requests
from bs4 import BeautifulSoup
import time
import numpy as np

class TickerManager:
    def __init__(self):
        self.data_dir = "data/exports"
        os.makedirs(self.data_dir, exist_ok=True)
        self.cache_file = os.path.join(self.data_dir, "tickers_cache.json")
        self.categories = self._initialize_categories()
        logging.info(f"TickerManager initialized with {self._count_total_tickers()} tickers")

    def _count_total_tickers(self) -> int:
        """Count total number of unique tickers across all categories."""
        all_tickers = set()
        for tickers in self.categories.values():
            all_tickers.update(tickers)
        return len(all_tickers)

    def _initialize_categories(self) -> Dict[str, List[str]]:
        """Initialize categories with cached data or fetch new data."""
        try:
            # Try to load from cache first
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r') as f:
                    cached_data = json.load(f)
                    if cached_data.get('timestamp', 0) > time.time() - 86400:  # 24 hour cache
                        logging.info("Loading tickers from cache")
                        return cached_data['categories']

            # If no cache or expired, fetch new data
            logging.info("Fetching fresh ticker data")
            categories = self._fetch_all_tickers()
            
            # Save to cache
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'timestamp': time.time(),
                    'categories': categories
                }, f)
            
            return categories

        except Exception as e:
            logging.error(f"Error initializing categories: {e}")
            return self._get_default_categories()

    def _get_default_categories(self) -> Dict[str, List[str]]:
        """Return default categories if fetching fails."""
        return {
            'Mixed Tickers': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'V', 'BAC', 'GS',
                'JNJ', 'PFE', 'UNH', 'MRK', 'WMT', 'PG', 'KO', 'MCD'
            ],
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD'],
            'Financial': ['JPM', 'BAC', 'GS', 'MS', 'V'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'ABBV']
        }

    def _fetch_all_tickers(self) -> Dict[str, List[str]]:
        """Fetch all tickers from NASDAQ and NYSE."""
        try:
            # Fetch NASDAQ tickers
            nasdaq_url = "https://www.nasdaq.com/market-activity/stocks/screener"
            nyse_url = "https://www.nyse.com/listings_directory/stock"

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            # Initialize categories
            categories = {
                'NASDAQ Large Cap': [],
                'NASDAQ Mid Cap': [],
                'NASDAQ Small Cap': [],
                'NYSE Large Cap': [],
                'NYSE Mid Cap': [],
                'NYSE Small Cap': [],
                'Technology': [],
                'Financial': [],
                'Healthcare': [],
                'Consumer': [],
                'Industrial': [],
                'Energy': [],
                'Mixed Popular': [
                    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
                    'JPM', 'V', 'BAC', 'GS', 'JNJ', 'PFE', 'UNH', 'MRK'
                ]
            }

            # Use yfinance's stock info for basic categorization
            tickers_info = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            
            for _, row in tickers_info.iterrows():
                ticker = row['Symbol']
                sector = row['GICS Sector']
                
                # Add to sector categories
                if 'Technology' in sector:
                    categories['Technology'].append(ticker)
                elif 'Financial' in sector:
                    categories['Financial'].append(ticker)
                elif 'Health' in sector:
                    categories['Healthcare'].append(ticker)
                elif 'Consumer' in sector:
                    categories['Consumer'].append(ticker)
                elif 'Industrial' in sector:
                    categories['Industrial'].append(ticker)
                elif 'Energy' in sector:
                    categories['Energy'].append(ticker)

            # Add some popular ETFs
            categories['ETFs'] = [
                'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO', 'VEA', 'VWO',
                'BND', 'AGG', 'GLD', 'SLV', 'VNQ', 'XLF', 'XLK', 'XLE'
            ]

            logging.info("Successfully fetched and categorized tickers")
            return categories

        except Exception as e:
            logging.error(f"Error fetching tickers: {e}")
            return self._get_default_categories()

    def get_categories(self) -> Dict[str, List[str]]:
        """Get all categories and their tickers."""
        return self.categories

    def get_category_names(self) -> List[str]:
        """Get list of category names."""
        return list(self.categories.keys())

    def get_tickers_in_category(self, category: str) -> List[str]:
        """Get tickers for a specific category."""
        return self.categories.get(category, [])

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

    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime, interval: str = '1d') -> Optional[pd.DataFrame]:
        """Get detailed historical data including technical indicators."""
        try:
            logging.info(f"Fetching historical data for {symbol}")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                auto_adjust=True
            )
            
            if df.empty:
                logging.warning(f"No historical data found for {symbol}")
                return None

            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Add basic info
            info = ticker.fast_info
            df['Symbol'] = symbol
            df['Market_Cap'] = info.market_cap if hasattr(info, 'market_cap') else 0
            
            return df

        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {e}")
            return None

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

    def get_quotes_df(self, symbols: List[str], start_date: datetime, 
                     end_date: datetime, interval: str = '1d') -> pd.DataFrame:
        """Get historical quotes with technical indicators for multiple symbols."""
        try:
            all_data = []
            total = len(symbols)
            
            for symbol in symbols:
                try:
                    df = self.get_historical_data(symbol, start_date, end_date, interval)
                    if df is not None:
                        # Reset index to make date a column
                        df = df.reset_index()
                        all_data.append(df)
                    
                    # Add delay to prevent rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    logging.error(f"Error processing {symbol}: {e}")
                    continue
            
            if not all_data:
                return pd.DataFrame()
                
            # Combine all data
            combined_df = pd.concat(all_data, axis=0)
            
            # Convert date column to string for easier export
            combined_df['Date'] = combined_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            return combined_df

        except Exception as e:
            logging.error(f"Error in get_quotes_df: {e}")
            return pd.DataFrame()

    def export_to_csv(self, df: pd.DataFrame, filename: str) -> str:
        """Export data to CSV."""
        try:
            path = os.path.join(self.data_dir, f"{filename}.csv")
            df.to_csv(path, index=False)
            return path
        except Exception as e:
            logging.error(f"Error exporting to CSV: {e}")
            raise

    def export_to_json(self, df: pd.DataFrame, filename: str) -> str:
        """Export data to JSON."""
        try:
            path = os.path.join(self.data_dir, f"{filename}.json")
            df.to_json(path, orient='records', indent=2)
            return path
        except Exception as e:
            logging.error(f"Error exporting to JSON: {e}")
            raise

    def export_to_sqlite(self, df: pd.DataFrame, filename: str, table_name: str) -> str:
        """Export data to SQLite."""
        try:
            path = os.path.join(self.data_dir, f"{filename}.db")
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
            path = os.path.join(self.data_dir, f"{filename}.duckdb")
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