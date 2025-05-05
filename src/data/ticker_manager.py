import pandas as pd
from pathlib import Path
import logging
import yfinance as yf
from datetime import datetime, timedelta
import os
import json
import sqlite3
import polars as pl
import duckdb
from typing import Dict, List, Union, Tuple
import time
from requests.exceptions import HTTPError
import asyncio
import aiohttp
import concurrent.futures
from functools import partial
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import pandas_datareader.data as pdr
except ImportError:
    pdr = None

logger = logging.getLogger(__name__)

class TickerManager:
    def __init__(self, db_name=None, db_dir=None):
        self.tickers = {}
        self.categories = ['All', 'Technology', 'Finance', 'Consumer', 'Healthcare', 'Industrial', 'Mixed']
        # Allow custom db_dir and db_name
        if db_dir is not None:
            self.data_path = Path(db_dir)
        else:
            self.data_path = Path('/Users/porupine/Documents/GitHub/stock-pred/data/stock_new/data')
        self.cache_dir = self.data_path / 'cache'
        self.exports_dir = self.data_path / 'exports'
        self.cache_dir.mkdir(exist_ok=True)
        self.exports_dir.mkdir(exist_ok=True)
        self.db_name = db_name
        
        # Define supported intervals
        self.supported_intervals = {
            '1d': '1d',    # Daily
            '1w': '1wk',   # Weekly
            '1mo': '1mo',  # Monthly
            '3mo': '3mo',  # Quarterly
            '1y': '1y',    # Yearly
            '5y': '5y',    # 5 Years
            'max': 'max'   # Maximum available
        }
        
        self.load_tickers()
        
        # Rate limiting parameters - extremely conservative settings
        self.requests_per_minute = 1  # Only 1 request per minute
        self.min_request_interval = 60 / self.requests_per_minute  # seconds between requests
        self.last_request_time = 0
        self.request_count = 0
        self.request_window_start = time.time()
        
        # Session-based rate limiting (for Yahoo Finance daily limits)
        self.max_session_requests = 10  # Reduced to 10 requests per session window
        self.session_window = 3600  # 1 hour session window
        self.session_request_count = 0
        self.session_start_time = time.time()
        
        # Ticker cooldown - increased time between different tickers
        self.ticker_cooldown = 180  # Increased to 180 seconds between different tickers
        self.last_ticker = None
        self.last_ticker_time = 0
        
        # Retry parameters - extremely conservative with more backoff
        self.max_retries = 1  # Only retry once
        self.base_delay = 120  # Increased to 120 seconds
        self.max_delay = 300  # Increased to 300 seconds
        self.jitter = 30  # Increased to 30 seconds
        
        # Async parameters - reduced concurrency
        self.max_concurrent_downloads = 1  # Only 1 at a time

    def _rate_limit(self):
        """Implement enhanced rate limiting with session tracking and ticker cooldown"""
        current_time = time.time()
        
        # Reset counter if we're in a new minute
        if current_time - self.request_window_start >= 60:
            self.request_count = 0
            self.request_window_start = current_time
        
        # Reset session counter if we're in a new session window
        if current_time - self.session_start_time >= self.session_window:
            self.session_request_count = 0
            self.session_start_time = current_time
        
        # Apply ticker-specific cooldown
        if self.last_ticker and current_time - self.last_ticker_time < self.ticker_cooldown:
            sleep_time = self.ticker_cooldown - (current_time - self.last_ticker_time)
            logger.info(f"Ticker cooldown active. Waiting {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        
        # Check if we've hit the rate limit
        if self.request_count >= self.requests_per_minute:
            sleep_time = 60 - (current_time - self.request_window_start)
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            self.request_count = 0
            self.request_window_start = time.time()
        
        # Check if we've hit the session limit
        if self.session_request_count >= self.max_session_requests:
            sleep_time = self.session_window - (current_time - self.session_start_time)
            if sleep_time > 0:
                logger.info(f"Session limit reached. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            self.session_request_count = 0
            self.session_start_time = time.time()
        
        # Ensure minimum interval between requests with jitter
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            # Add random jitter
            sleep_time += random.uniform(0, self.jitter)
            logger.info(f"Rate limiting: waiting {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
        self.session_request_count += 1

    def _handle_yfinance_error(self, ticker: str, error: Exception, attempt: int) -> bool:
        """Handle Yahoo Finance API errors with exponential backoff and jitter"""
        if isinstance(error, HTTPError):
            if error.response.status_code == 429:  # Too Many Requests
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    # Add random jitter
                    delay += random.uniform(0, self.jitter)
                    logger.warning(f"Rate limit hit for {ticker}, waiting {delay:.2f} seconds...")
                    time.sleep(delay)
                    return True
                else:
                    logger.error(f"Max retries reached for {ticker}")
                    return False
            elif error.response.status_code == 404:
                logger.error(f"Ticker {ticker} not found")
                return False
            elif error.response.status_code == 503:
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    # Add random jitter
                    delay += random.uniform(0, self.jitter)
                    logger.warning(f"Service unavailable for {ticker}, retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                    return True
                else:
                    logger.error(f"Service unavailable for {ticker} after {self.max_retries} attempts")
                    return False
            else:
                logger.error(f"HTTP error for {ticker}: {error}")
                return False
        elif isinstance(error, ConnectionError):
            if attempt < self.max_retries:
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                # Add random jitter
                delay += random.uniform(0, self.jitter)
                logger.warning(f"Connection error for {ticker}, retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                return True
            else:
                logger.error(f"Max retries reached for {ticker}")
                return False
        elif isinstance(error, ValueError) and "Expecting value" in str(error):
            if attempt < self.max_retries:
                delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                # Add random jitter
                delay += random.uniform(0, self.jitter)
                logger.warning(f"Invalid response for {ticker}, retrying in {delay:.2f} seconds...")
                time.sleep(delay)
                return True
            else:
                logger.error(f"Invalid response for {ticker} after {self.max_retries} attempts")
                return False
        else:
            logger.error(f"Unexpected error for {ticker}: {error}")
            return False

    def load_tickers(self):
        """Load tickers with fallback mechanism"""
        try:
            # Initialize categories
            self.tickers = {category: [] for category in self.categories}
            
            # Try to load from file
            file_path = self.data_path / 'nasdaq_screener.csv'
            
            if not file_path.exists():
                logger.error(f"Ticker file not found: {file_path}")
                self._create_default_tickers()
                return

            df = pd.read_csv(file_path)
            
            # Process each ticker
            for _, row in df.iterrows():
                symbol = row['Symbol']
                sector = row['Sector']
                
                if not isinstance(symbol, str) or not symbol.strip():
                    logger.warning(f"Non-string or empty symbol found: {symbol} (type: {type(symbol)})")
                    continue
                
                # Add to sector category
                if sector in self.tickers:
                    self.tickers[sector].append(symbol)
                
                # Add to 'All' category
                if symbol not in self.tickers['All']:
                    self.tickers['All'].append(symbol)
            
            # Create mixed categories
            self._create_mixed_categories()
            
            logger.info(f"Successfully loaded {len(self.tickers['All'])} tickers")
            
        except Exception as e:
            logger.error(f"Error loading tickers: {e}")
            self._create_default_tickers()

    async def _download_ticker_async(self, ticker: str, start_date: str, end_date: str, interval: str = '1d', semaphore=None) -> Tuple[str, pd.DataFrame]:
        """Download data for a single ticker asynchronously"""
        if semaphore is None:
            # fallback to no concurrency limit if not provided
            class DummySemaphore:
                async def __aenter__(self): return None
                async def __aexit__(self, exc_type, exc, tb): return None
            semaphore = DummySemaphore()
        async with semaphore:
            # Apply rate limiting
            self._rate_limit()
            # Apply ticker cooldown
            current_time = time.time()
            if self.last_ticker and self.last_ticker != ticker and current_time - self.last_ticker_time < self.ticker_cooldown:
                sleep_time = self.ticker_cooldown - (current_time - self.last_ticker_time)
                logger.info(f"Async ticker cooldown for {ticker}. Waiting {sleep_time:.2f} seconds...")
                await asyncio.sleep(sleep_time)
            # Update last ticker info
            self.last_ticker = ticker
            self.last_ticker_time = time.time()
            try:
                # Check for cached data
                cache_file = self.cache_dir / f"{ticker}_{start_date}_{end_date}_{interval}.csv"
                if cache_file.exists():
                    logger.info(f"Using cached data for {ticker}")
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                    df.index = pd.to_datetime(df.index, utc=True)
                    return ticker, df
                # Get the current event loop
                loop = asyncio.get_running_loop()
                # Create a new event loop for the thread pool executor
                executor_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(executor_loop)
                try:
                    # Run the download in the thread pool executor
                    result = await loop.run_in_executor(
                        None, 
                        lambda: self._download_ticker_parallel(ticker, start_date, end_date, interval)
                    )
                    if result is not None:
                        # Unpack the tuple returned by _download_ticker_parallel
                        result_ticker, df = result
                        return result_ticker, df
                    return ticker, None
                finally:
                    # Clean up the executor loop
                    executor_loop.close()
            except Exception as e:
                logger.error(f"Async error downloading {ticker}: {e}")
                return ticker, None

    def _download_ticker_parallel(self, ticker, start_date, end_date, interval, data_source='yfinance', pdr_sub_source='stooq'):
        """Download data for a single ticker (for parallel or sequential processing)"""
        try:
            logger.info(f"Downloading data for {ticker} ({interval}) from {start_date} to {end_date} using {data_source}")
            # Apply rate limiting
            self._rate_limit()
            # Apply ticker cooldown if needed
            current_time = time.time()
            if self.last_ticker and self.last_ticker != ticker and current_time - self.last_ticker_time < self.ticker_cooldown:
                sleep_time = self.ticker_cooldown - (current_time - self.last_ticker_time)
                logger.info(f"Applying ticker cooldown for {ticker}. Waiting {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            # Update last ticker info
            self.last_ticker = ticker
            self.last_ticker_time = time.time()
            # Check if data is already cached
            cache_file = self.cache_dir / f"{ticker}_{start_date}_{end_date}_{interval}_{data_source}.csv"
            if cache_file.exists():
                logger.info(f"Using cached data for {ticker}")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                df.index = pd.to_datetime(df.index, utc=True)
                return ticker, df
            # Download data using selected data source
            if data_source == 'yfinance':
                stock = yf.Ticker(ticker)
                try:
                    info = stock.info
                    if not info or 'regularMarketPrice' not in info:
                        logger.error(f"Ticker {ticker} appears to be invalid")
                        return ticker, None
                except Exception as e:
                    logger.error(f"Error getting info for {ticker}: {e}")
                    return ticker, None
                df = stock.history(start=start_date, end=end_date, interval=interval)
            elif data_source == 'pandas_datareader':
                if pdr is None:
                    logger.error("pandas_datareader is not installed.")
                    return ticker, None
                if interval != '1d':
                    logger.warning(f"pandas_datareader only supports daily data. Forcing interval to '1d'.")
                try:
                    df = pdr.DataReader(ticker, pdr_sub_source, start=start_date, end=end_date)
                except Exception as e:
                    logger.error(f"pandas_datareader error for {ticker}: {e}")
                    return ticker, None
            else:
                logger.error(f"Unknown data source: {data_source}")
                return ticker, None
            if df is None or df.empty:
                logger.error(f"No data available for {ticker}")
                return ticker, None
            # Save to cache
            try:
                df.to_csv(cache_file)
                logger.info(f"Successfully downloaded and cached data for {ticker}")
            except Exception as e:
                logger.warning(f"Failed to cache data for {ticker}: {e}")
            return ticker, df
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            delay = min(self.base_delay * (2 ** 0), self.max_delay)
            delay += random.uniform(0, self.jitter)
            logger.info(f"Waiting {delay:.2f} seconds before retry...")
            time.sleep(delay)
            return ticker, None

    async def get_historical_data_async(self, tickers: List[str], start_date: str, end_date: str, interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Get historical data asynchronously with batch processing"""
        try:
            data = {}
            logger.info(f"Using async download mode")
            # Use a smaller batch size for better rate limit management
            batch_size = 3  # Process 3 tickers at a time
            semaphore = asyncio.Semaphore(self.max_concurrent_downloads)
            # Process tickers in smaller batches
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i+batch_size]
                logger.info(f"Processing async batch {i//batch_size + 1} of {(len(tickers) + batch_size - 1) // batch_size} ({len(batch)} tickers)")
                # Create tasks for the current batch
                tasks = [self._download_ticker_async(ticker, start_date, end_date, interval, semaphore) for ticker in batch]
                # Wait for all tasks in this batch to complete
                results = await asyncio.gather(*tasks)
                # Process results
                for ticker, df in results:
                    if df is not None:
                        data[ticker] = df
                # Add a delay between batches
                if i + batch_size < len(tickers):
                    delay = self.base_delay/2 + random.uniform(0, self.jitter/2)
                    logger.info(f"Async batch complete. Waiting {delay:.2f} seconds before next batch...")
                    await asyncio.sleep(delay)
            logger.info(f"Download complete. Successfully downloaded data for {len(data)}/{len(tickers)} tickers")
            return data
        except Exception as e:
            logger.error(f"Error in async download: {e}")
            return {}

    def get_historical_data_parallel(self, tickers: List[str], start_date: str, end_date: str, interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """Get historical data in parallel with enhanced rate limiting"""
        try:
            data = {}
            
            # Use a smaller batch size to reduce concurrent load
            batch_size = 2  # Process just 2 tickers at a time
            
            # Process tickers in smaller batches
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1} of {(len(tickers) + batch_size - 1) // batch_size} ({len(batch)} tickers)")
                
                # Create a thread pool with limited concurrency
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    # Submit download tasks for the batch
                    futures = [executor.submit(self._download_ticker_parallel, ticker, start_date, end_date, interval) for ticker in batch]
                    
                    # Process completed downloads
                    for future, ticker in zip(futures, batch):
                        try:
                            result = future.result()
                            if result is not None:
                                data[ticker] = result
                        except Exception as e:
                            logger.error(f"Error in parallel download for {ticker}: {e}")
                
                # Add a delay between batches to avoid rate limits
                if i + batch_size < len(tickers):
                    delay = self.base_delay + random.uniform(0, self.jitter)
                    logger.info(f"Batch complete. Waiting {delay:.2f} seconds before next batch...")
                    time.sleep(delay)
            
            return data
            
        except Exception as e:
            logger.error(f"Error in parallel download: {e}")
            return {}

    def get_historical_data(self, tickers: List[str], start_date: str, end_date: str, 
                          interval: str = '1d', clean_data: bool = True, 
                          download_mode: str = 'sequential', data_source: str = 'yfinance',
                          pdr_sub_source: str = 'stooq') -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple tickers with improved rate limiting"""
        if not tickers:
            logger.error("No tickers provided")
            return {}
        
        logger.info(f"Starting data download for {len(tickers)} tickers in {download_mode} mode")
        logger.info(f"Date range: {start_date} to {end_date}, interval: {interval}, data_source: {data_source}")
        
        # Validate interval
        if interval not in self.supported_intervals:
            logger.error(f"Unsupported interval: {interval}")
            return {}
        
        # Convert interval to Yahoo Finance format
        yf_interval = self.supported_intervals[interval]
        
        # Initialize results dictionary
        results = {}
        
        # Process tickers based on download mode
        if download_mode == 'sequential':
            logger.info("Using sequential download mode")
            for i, ticker in enumerate(tickers, 1):
                logger.info(f"Processing {ticker} ({i}/{len(tickers)})")
                
                # Check cache first
                cache_file = self.cache_dir / f"{ticker}_{start_date}_{end_date}_{yf_interval}_{data_source}.csv"
                if cache_file.exists():
                    logger.info(f"Loading cached data for {ticker}")
                    try:
                        df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                        if clean_data:
                            df = self.clean_data(df)
                        results[ticker] = df
                        continue
                    except Exception as e:
                        logger.error(f"Error loading cached data for {ticker}: {e}")
                
                # Download data
                result_ticker, df = self._download_ticker_parallel(
                    ticker, start_date, end_date, interval, data_source, pdr_sub_source
                )
                if df is not None:
                    if clean_data:
                        df = self.clean_data(df)
                    results[result_ticker] = df
                
                # Add cool-down period between tickers
                if i < len(tickers):
                    cool_down = self.ticker_cooldown + random.uniform(0, 10)
                    logger.info(f"Cool-down period: waiting {cool_down:.2f} seconds before next ticker...")
                    time.sleep(cool_down)
        
        elif download_mode == 'parallel':
            logger.info("Using parallel download mode")
            with ThreadPoolExecutor(max_workers=self.max_concurrent_downloads) as executor:
                futures = []
                for ticker in tickers:
                    future = executor.submit(
                        self._download_ticker_parallel, ticker, start_date, end_date, interval, data_source, pdr_sub_source
                    )
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        result_ticker, df = future.result()
                        if df is not None:
                            if clean_data:
                                df = self.clean_data(df)
                            results[result_ticker] = df
                    except Exception as e:
                        logger.error(f"Error handling future result: {e}")
                        continue
        
        elif download_mode == 'async':
            logger.info("Using async download mode")
            async def download_all():
                tasks = []
                for ticker in tickers:
                    task = self._download_ticker_async(ticker, start_date, end_date, interval)
                    tasks.append(task)
                return await asyncio.gather(*tasks)
            
            results_list = asyncio.run(download_all())
            for result_ticker, df in results_list:
                if df is not None:
                    if clean_data:
                        df = self.clean_data(df)
                    results[result_ticker] = df
        
        else:
            logger.error(f"Unsupported download mode: {download_mode}")
            return {}
        
        # Log results
        success_count = len(results)
        logger.info(f"Download complete. Successfully downloaded data for {success_count}/{len(tickers)} tickers")
        
        return results

    def get_date_ranges(self):
        """Get predefined date ranges for data download"""
        today = datetime.now()
        return {
            '1 Month': (today - timedelta(days=30)).strftime('%Y-%m-%d'),
            '3 Months': (today - timedelta(days=90)).strftime('%Y-%m-%d'),
            '6 Months': (today - timedelta(days=180)).strftime('%Y-%m-%d'),
            '1 Year': (today - timedelta(days=365)).strftime('%Y-%m-%d'),
            '2 Years': (today - timedelta(days=730)).strftime('%Y-%m-%d'),
            '5 Years': (today - timedelta(days=1825)).strftime('%Y-%m-%d')
        }

    def clear_cache(self):
        """Clear the data cache"""
        try:
            for file in self.cache_dir.glob('*.csv'):
                file.unlink()
            logger.info("Cleared data cache")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")

    def _create_mixed_categories(self):
        """Create mixed categories by combining tickers from different sectors"""
        try:
            # Tech + Finance mix
            tech_finance = list(set(self.tickers['Technology'][:5] + self.tickers['Finance'][:5]))
            
            # Tech + Healthcare mix
            tech_health = list(set(self.tickers['Technology'][:5] + self.tickers['Healthcare'][:5]))
            
            # Consumer + Industrial mix
            consumer_industrial = list(set(self.tickers['Consumer'][:5] + self.tickers['Industrial'][:5]))
            
            # Combine all mixed categories
            self.tickers['Mixed'] = list(set(tech_finance + tech_health + consumer_industrial))
            
            logger.info(f"Created mixed category with {len(self.tickers['Mixed'])} tickers")
            
        except Exception as e:
            logger.error(f"Error creating mixed categories: {e}")
            self.tickers['Mixed'] = []

    def _create_default_tickers(self):
        """Create default tickers if loading fails"""
        default_tickers = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
            'Finance': ['JPM', 'BAC', 'GS', 'MS', 'V'],
            'Consumer': ['AMZN', 'WMT', 'HD', 'MCD', 'NKE'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
            'Industrial': ['BA', 'CAT', 'GE', 'HON', 'MMM'],
            'Mixed': ['AAPL', 'JPM', 'AMZN', 'JNJ', 'BA', 'MSFT', 'BAC', 'WMT', 'PFE', 'CAT']
        }
        
        self.tickers = {category: [] for category in self.categories}
        for category, symbols in default_tickers.items():
            self.tickers[category] = symbols.copy()
            self.tickers['All'].extend(symbols)
        
        # Remove duplicates from 'All' category
        self.tickers['All'] = list(dict.fromkeys(self.tickers['All']))
        logger.info("Created default tickers as fallback")

    def get_all_tickers(self):
        """Get all tickers"""
        # Only include non-empty strings
        return sorted([str(t) for t in set(self.tickers['All']) if isinstance(t, str) and t.strip()])

    def get_categories(self):
        """Get all categories"""
        return self.categories

    def get_tickers_by_category(self, category):
        """Get tickers for a specific category"""
        return sorted([str(t) for t in set(self.tickers.get(category, [])) if isinstance(t, str) and t.strip()])

    def get_ticker_info(self, symbol):
        """Get information about a specific ticker"""
        try:
            file_path = self.data_path / 'nasdaq_screener.csv'
            df = pd.read_csv(file_path)
            row = df[df['Symbol'] == symbol].iloc[0]
            return {
                'Symbol': row['Symbol'],
                'Name': row['Security Name'],
                'Category': row['Sector']
            }
        except Exception:
            return None

    def search_tickers(self, search_text):
        """Search for tickers by symbol or name"""
        try:
            file_path = self.data_path / 'nasdaq_screener.csv'
            df = pd.read_csv(file_path)
            search_text = search_text.upper()
            mask = (df['Symbol'].str.upper().str.contains(search_text)) | \
                   (df['Security Name'].str.upper().str.contains(search_text))
            return sorted(df[mask]['Symbol'].tolist())
        except Exception:
            return []

    def export_data(self, data, format='csv', filename=None):
        """Export data to specified format"""
        if not data:
            raise ValueError("No data to export")

        # Create export directory if it doesn't exist
        export_dir = self.get_export_dir()
        os.makedirs(export_dir, exist_ok=True)

        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"stock_data_{timestamp}"

        if format == 'csv':
            # Export each ticker to a separate CSV file
            for ticker, df in data.items():
                # Convert timestamps to strings
                df_copy = df.copy()
                if isinstance(df_copy.index, pd.DatetimeIndex):
                    df_copy.index = df_copy.index.strftime('%Y-%m-%d %H:%M:%S')
                filepath = os.path.join(export_dir, f"{filename}_{ticker}.csv")
                df_copy.to_csv(filepath)
            return export_dir

        elif format == 'json':
            # Convert data to JSON-serializable format
            json_data = {}
            for ticker, df in data.items():
                # Convert timestamps to strings and DataFrame to dict
                df_copy = df.copy()
                if isinstance(df_copy.index, pd.DatetimeIndex):
                    df_copy.index = df_copy.index.strftime('%Y-%m-%d %H:%M:%S')
                json_data[ticker] = df_copy.to_dict(orient='index')
            
            filepath = os.path.join(export_dir, f"{filename}.json")
            with open(filepath, 'w') as f:
                json.dump(json_data, f, indent=4)
            return filepath

        elif format == 'excel':
            # Export all data to a single Excel file with multiple sheets
            filepath = os.path.join(export_dir, f"{filename}.xlsx")
            with pd.ExcelWriter(filepath) as writer:
                for ticker, df in data.items():
                    # Convert timestamps to strings
                    df_copy = df.copy()
                    if isinstance(df_copy.index, pd.DatetimeIndex):
                        df_copy.index = df_copy.index.strftime('%Y-%m-%d %H:%M:%S')
                    df_copy.to_excel(writer, sheet_name=ticker)
            return filepath
            
        elif format == 'sqlite':
            # Export to SQLite database
            filepath = os.path.join(export_dir, f"{filename}.db")
            conn = sqlite3.connect(filepath)
            for ticker, df in data.items():
                df_copy = df.copy()
                if isinstance(df_copy.index, pd.DatetimeIndex):
                    df_copy = df_copy.reset_index()
                    df_copy.rename(columns={'index': 'date'}, inplace=True)
                df_copy.to_sql(f"stock_{ticker}", conn, if_exists='replace', index=False)
            conn.close()
            return filepath
            
        elif format == 'duckdb':
            # Export to DuckDB database
            filepath = os.path.join(export_dir, f"{filename}.duckdb")
            conn = duckdb.connect(filepath)
            for ticker, df in data.items():
                df_copy = df.copy()
                if isinstance(df_copy.index, pd.DatetimeIndex):
                    df_copy = df_copy.reset_index()
                    df_copy.rename(columns={'index': 'date'}, inplace=True)
                conn.execute(f"CREATE TABLE IF NOT EXISTS stock_{ticker} AS SELECT * FROM df_copy")
            conn.close()
            return filepath
            
        elif format == 'polars':
            # Convert to Polars DataFrames
            polars_data = {}
            for ticker, df in data.items():
                polars_data[ticker] = self.convert_to_polars(df)
            # Save in Parquet format as a sample output
            filepath = os.path.join(export_dir, f"{filename}.parquet")
            # Combine all dataframes into one with a 'ticker' column
            combined_df = None
            for ticker, pl_df in polars_data.items():
                pl_df = pl_df.with_column(pl.lit(ticker).alias('ticker'))
                if combined_df is None:
                    combined_df = pl_df
                else:
                    combined_df = combined_df.vstack(pl_df)
            if combined_df is not None:
                combined_df.write_parquet(filepath)
            return filepath
            
        elif format == 'parquet':
            # Export to Parquet format
            filepath = os.path.join(export_dir, f"{filename}.parquet")
            combined_df = None
            for ticker, df in data.items():
                df_copy = df.copy()
                df_copy['ticker'] = ticker
                if combined_df is None:
                    combined_df = df_copy
                else:
                    combined_df = pd.concat([combined_df, df_copy])
            if combined_df is not None:
                combined_df.to_parquet(filepath, index=True)
            return filepath
            
        elif format == 'arrow':
            # Export to Arrow format
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq
                
                filepath = os.path.join(export_dir, f"{filename}.arrow")
                combined_data = []
                for ticker, df in data.items():
                    df_copy = df.copy()
                    df_copy['ticker'] = ticker
                    combined_data.append(df_copy)
                
                if combined_data:
                    combined_df = pd.concat(combined_data)
                    table = pa.Table.from_pandas(combined_df)
                    pq.write_table(table, filepath)
                return filepath
            except ImportError:
                raise ValueError("pyarrow package is required for Arrow format")

        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_export_formats(self) -> List[str]:
        """Get list of supported export formats"""
        return ['csv', 'json', 'excel', 'sqlite', 'duckdb', 'polars', 'parquet', 'arrow']

    def get_export_dir(self) -> str:
        """Get the exports directory path"""
        return str(self.exports_dir)

    def clean_data(self, df: pd.DataFrame, options: Dict[str, bool] = None) -> pd.DataFrame:
        """
        Clean the stock data DataFrame
        
        Args:
            df (pd.DataFrame): Input DataFrame with stock data
            options (dict): Dictionary of cleaning options:
                - handle_missing: Handle missing values
                - remove_outliers: Remove price outliers
                - standardize_dates: Ensure consistent date format
                - fill_gaps: Fill missing trading days
                - normalize_volume: Normalize volume data
                - remove_duplicates: Remove duplicate entries
                
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if options is None:
            options = {
                'handle_missing': True,
                'remove_outliers': True,
                'standardize_dates': True,
                'fill_gaps': True,
                'normalize_volume': True,
                'remove_duplicates': True
            }
        
        try:
            # Create a copy to avoid modifying the original
            cleaned_df = df.copy()
            
            # Handle missing values
            if options.get('handle_missing', True):
                cleaned_df = self.handle_missing_values(cleaned_df)
            
            # Remove outliers
            if options.get('remove_outliers', True):
                cleaned_df = self._remove_outliers(cleaned_df)
            
            # Standardize dates
            if options.get('standardize_dates', True):
                cleaned_df = self.standardize_dates(cleaned_df)
            
            # Fill missing trading days
            if options.get('fill_gaps', True):
                cleaned_df = self._fill_trading_gaps(cleaned_df)
            
            # Normalize volume
            if options.get('normalize_volume', True):
                cleaned_df = self._normalize_volume(cleaned_df)
            
            # Remove duplicates
            if options.get('remove_duplicates', True):
                cleaned_df = self._remove_duplicates(cleaned_df)
            
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Error cleaning data: {e}")
            return df  # Return original DataFrame if cleaning fails

    def handle_missing_values(self, df):
        """Handle missing values in the DataFrame"""
        try:
            # Forward fill missing values
            df = df.ffill()
            # Backward fill any remaining missing values
            df = df.bfill()
            return df
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            return df

    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove price outliers using the Interquartile Range (IQR) method"""
        try:
            # Calculate IQR for price columns
            price_columns = ['Open', 'High', 'Low', 'Close']
            for col in price_columns:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 3 * IQR
                    upper_bound = Q3 + 3 * IQR
                    
                    # Replace outliers with NaN
                    df[col] = df[col].apply(lambda x: x if lower_bound <= x <= upper_bound else None)
            
            # Handle the NaN values created by outlier removal
            df = self.handle_missing_values(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error removing outliers: {e}")
            return df

    def standardize_dates(self, df):
        """Standardize date format and timezone"""
        try:
            # Convert index to datetime if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            
            # Handle timezone if present
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_convert('UTC')
            elif hasattr(df.index, 'tz') and df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            return df
        except Exception as e:
            logger.error(f"Error standardizing dates: {e}")
            return df

    def _fill_trading_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing trading days with forward-filled values"""
        try:
            # Ensure index is datetime with UTC timezone
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, utc=True)
            elif df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            
            # Create a complete date range in UTC
            date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B', tz='UTC')
            
            # Reindex the DataFrame
            df = df.reindex(date_range)
            
            # Forward fill missing values
            df = df.ffill()
            
            return df
            
        except Exception as e:
            logger.error(f"Error filling trading gaps: {e}")
            return df

    def _normalize_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize volume data to handle large variations"""
        try:
            if 'Volume' in df.columns:
                # Calculate rolling average volume
                df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
                
                # Calculate volume ratio
                df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
                
                # Remove the temporary MA column
                df = df.drop('Volume_MA', axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error normalizing volume: {e}")
            return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate entries from the DataFrame"""
        try:
            # Remove duplicates based on index
            df = df[~df.index.duplicated(keep='first')]
            
            return df
            
        except Exception as e:
            logger.error(f"Error removing duplicates: {e}")
            return df

    def save_data(self, data: Dict[str, pd.DataFrame], filename: str = None) -> str:
        """Save data to CSV files"""
        try:
            if not data:
                logger.error("No data to save")
                return None

            # Create export directory if it doesn't exist
            self.exports_dir.mkdir(exist_ok=True)
            
            # Generate filename if not provided
            if not filename:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"stock_data_{timestamp}"
            
            saved_files = []
            
            for ticker, df in data.items():
                if df is not None and not df.empty:
                    try:
                        # Create a copy of the DataFrame to avoid modifying the original
                        df_copy = df.copy()
                        
                        # Ensure the DataFrame has the correct index with UTC timezone
                        if not isinstance(df_copy.index, pd.DatetimeIndex):
                            # Try to convert to DatetimeIndex, but don't use strftime if it's not
                            try:
                                df_copy.index = pd.to_datetime(df_copy.index, utc=True)
                            except Exception as e:
                                logger.warning(f"Could not convert index to DatetimeIndex for {ticker}: {e}")
                                # Continue without conversion, will write original index
                        elif df_copy.index.tz is None:
                            df_copy.index = df_copy.index.tz_localize('UTC')
                        
                        # Convert index to string format for saving only if it's a DatetimeIndex
                        if isinstance(df_copy.index, pd.DatetimeIndex):
                            df_copy.index = df_copy.index.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Create filepath
                        filepath = self.exports_dir / f"{filename}_{ticker}.csv"
                        
                        # Save to CSV
                        df_copy.to_csv(filepath)
                        saved_files.append(str(filepath))
                        logger.info(f"Successfully saved data for {ticker} to {filepath}")
                    except Exception as e:
                        logger.error(f"Error saving data for {ticker}: {e}")
                        continue

            if saved_files:
                logger.info(f"Successfully saved data for {len(saved_files)} tickers")
                return str(self.exports_dir)
            else:
                logger.error("No data was saved successfully")
                return None

        except Exception as e:
            logger.error(f"Error in save_data: {e}")
            return None

    def export_to_excel(self, df: pd.DataFrame, filename: str) -> str:
        """Export data to Excel (XLSX)."""
        try:
            path = self.exports_dir / f"{filename}.xlsx"
            df.to_excel(path, index=False)
            return path
        except Exception as e:
            logging.error(f"Error exporting to Excel: {e}")
            raise

    def export_to_parquet(self, df: pd.DataFrame, filename: str) -> str:
        """Export data to Parquet format."""
        try:
            path = self.exports_dir / f"{filename}.parquet"
            df.to_parquet(path, index=False)
            return path
        except Exception as e:
            logging.error(f"Error exporting to Parquet: {e}")
            raise

    def export_to_arrow(self, df: pd.DataFrame, filename: str) -> str:
        """Export data to Arrow format."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
            
            path = self.exports_dir / f"{filename}.arrow"
            table = pa.Table.from_pandas(df)
            pq.write_table(table, path)
            return path
        except Exception as e:
            logging.error(f"Error exporting to Arrow: {e}")
            raise

    def export_to_hdf5(self, df: pd.DataFrame, filename: str) -> str:
        """Export data to HDF5 format."""
        try:
            path = self.exports_dir / f"{filename}.h5"
            df.to_hdf(path, key='stock_data', mode='w')
            return path
        except Exception as e:
            logging.error(f"Error exporting to HDF5: {e}")
            raise

    def export_to_csv(self, df: pd.DataFrame, filename: str) -> str:
        """Export data to CSV."""
        try:
            # Create a copy of the DataFrame to avoid modifying the original
            df_copy = df.copy()
            
            # Handle DatetimeIndex if present
            if isinstance(df_copy.index, pd.DatetimeIndex):
                # Convert DatetimeIndex to string
                df_copy.index = df_copy.index.strftime('%Y-%m-%d %H:%M:%S')
            
            path = self.exports_dir / f"{filename}.csv"
            df_copy.to_csv(path)
            return str(path)
        except Exception as e:
            logger.error(f"Error exporting to CSV: {e}")
            raise

    def export_to_json(self, df: pd.DataFrame, filename: str) -> str:
        """Export data to JSON."""
        try:
            # Create a copy of the DataFrame to avoid modifying the original
            df_copy = df.copy()
            
            # Handle DatetimeIndex if present
            if isinstance(df_copy.index, pd.DatetimeIndex):
                # Convert DatetimeIndex to string
                df_copy.index = df_copy.index.strftime('%Y-%m-%d %H:%M:%S')
            
            path = self.exports_dir / f"{filename}.json"
            df_copy.to_json(path, orient='records', date_format='iso')
            return str(path)
        except Exception as e:
            logger.error(f"Error exporting to JSON: {e}")
            raise

    def export_to_sqlite(self, df: pd.DataFrame, filename: str, table_name: str) -> str:
        """Export data to SQLite."""
        try:
            # Create a copy of the DataFrame to avoid modifying the original
            df_copy = df.copy()
            
            # Handle DatetimeIndex by resetting index to a column
            if isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy = df_copy.reset_index()
                df_copy.rename(columns={'index': 'date'}, inplace=True)
            
            path = self.exports_dir / f"{filename}.db"
            conn = sqlite3.connect(str(path))
            df_copy.to_sql(table_name, conn, if_exists='replace', index=False)
            conn.close()
            return str(path)
        except Exception as e:
            logger.error(f"Error exporting to SQLite: {e}")
            raise

    def export_to_duckdb(self, df: pd.DataFrame, filename: str, table_name: str) -> str:
        """Export data to DuckDB."""
        try:
            # Create a copy of the DataFrame to avoid modifying the original
            df_copy = df.copy()
            
            # Handle DatetimeIndex by resetting index to a column
            if isinstance(df_copy.index, pd.DatetimeIndex):
                df_copy = df_copy.reset_index()
                df_copy.rename(columns={'index': 'date'}, inplace=True)
            
            path = self.exports_dir / f"{filename}.duckdb"
            conn = duckdb.connect(str(path))
            conn.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM df_copy")
            conn.close()
            return str(path)
        except Exception as e:
            logger.error(f"Error exporting to DuckDB: {e}")
            raise

    def convert_to_polars(self, df: pd.DataFrame) -> pl.DataFrame:
        """Convert pandas DataFrame to polars DataFrame."""
        try:
            # Create polars DataFrame from pandas
            pl_df = pl.from_pandas(df)
            return pl_df
        except Exception as e:
            logger.error(f"Error converting to Polars: {e}")
            raise 