import yfinance as yf
import pandas as pd
import sqlite3
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
from typing import List, Optional, Dict
from datetime import datetime
from database.database_connector import DatabaseConnector

class DataCollector:
    def __init__(self, config_path: str = "config.json", cache_config_path: str = "cache_config.json"):
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        self.cache_config = self._load_config(cache_config_path)
        
        # Create directories if they don't exist
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        Path("data/clean").mkdir(parents=True, exist_ok=True)
        Path("cache").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # Initialize cache connection
        self.cache_db = sqlite3.connect("cache/realtime_cache.db")
        self._initialize_cache_table()
        self.db_connector = DatabaseConnector()

    def _load_config(self, config_path: str) -> Dict:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"{config_path} not found, using defaults")
            return {}

    def _initialize_cache_table(self):
        """Initialize the SQLite cache table with proper schema"""
        try:
            with self.cache_db:
                self.cache_db.execute("""
                    CREATE TABLE IF NOT EXISTS realtime_cache (
                        ticker TEXT,
                        timestamp DATETIME,
                        price REAL,
                        volume INTEGER,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (ticker, timestamp)
                    )
                """)
                # Create index for faster lookups
                self.cache_db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ticker_timestamp 
                    ON realtime_cache(ticker, timestamp)
                """)
        except Exception as e:
            self.logger.error(f"Error initializing cache table: {e}")
            raise

    def collect_historical_data(self, tickers: List[str], start_date: str, end_date: str) -> None:
        """Collect historical data for multiple tickers in parallel"""
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for ticker in tickers:
                futures.append(
                    executor.submit(self._download_ticker_data, ticker, start_date, end_date)
                )
            
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error in historical data collection: {e}")

    def _download_ticker_data(self, ticker: str, start_date: str, end_date: str) -> None:
        """Download data for a single ticker and save to raw CSV"""
        try:
            self.logger.info(f"Downloading data for {ticker}")
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if data.empty:
                self.logger.warning(f"No data received for {ticker}")
                return
                
            # Add ticker column and reset index
            data = data.reset_index()
            data['ticker'] = ticker
            
            # Save to raw CSV
            output_path = f"data/raw/raw_{ticker}.csv"
            data.to_csv(output_path, index=False)
            self.logger.info(f"Saved raw data for {ticker} to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error downloading {ticker}: {e}")
            raise

    def download_ticker_data(self, ticker: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """Download historical data for a single ticker"""
        try:
            self.logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}")
            
            # Download data from yfinance
            data = yf.download(ticker, start=start_date, end=end_date)
            
            if data.empty:
                self.logger.warning(f"No data received for {ticker}")
                return None
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Ensure 'Adj Close' column exists
            if 'Adj Close' not in data.columns:
                data['Adj Close'] = data['Close']
            
            # Log the data structure
            self.logger.info(f"Downloaded data columns: {data.columns.tolist()}")
            self.logger.info(f"Data shape: {data.shape}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error downloading {ticker}: {e}")
            raise

    def save_ticker_data(self, ticker: str, data: pd.DataFrame) -> bool:
        """Save ticker data to database"""
        try:
            return self.db_connector.save_ticker_data(ticker, data)
        except Exception as e:
            self.logger.error(f"Error saving data for {ticker}: {e}")
            return False

    def cleanup(self):
        """Clean up resources"""
        try:
            self.cache_db.close()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}") 