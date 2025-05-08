import logging
from typing import Optional, List
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

from modules.database import DatabaseConnector
from config.config_manager import ConfigurationManager

class DataLoader:
    """Handles loading and managing stock data."""
    
    def __init__(self, config, db=None):
        self.logger = logging.getLogger("DataLoader")
        self.config = config
        self.period = config.get('period', '2y')
        self.interval = config.get('interval', '1d')
        self._stop_realtime = False
        self.db = db  # Store database reference for later use
    
    def collect_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Collect historical data for a given ticker."""
        try:
            self.logger.info(f"Collecting historical data for {symbol}")
            
            # Create Ticker object
            ticker = yf.Ticker(symbol)
            
            # Get historical data using configuration
            df = ticker.history(
                period=self.period,
                interval=self.interval
            )
            
            if df.empty:
                self.logger.warning(f"No data available for {symbol}")
                return None
            
            # Process DataFrame
            df = self._process_dataframe(df, symbol)
            
            self.logger.info(f"Successfully collected {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting historical data for {symbol}: {e}")
            return None
    
    def _process_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Process the downloaded DataFrame."""
        # Reset index to make date a column
        df = df.reset_index()
        
        # Rename columns to match database schema
        column_mapping = {
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        df = df.rename(columns=column_mapping)
        
        # Add ticker column
        df['ticker'] = symbol
        
        # Use 'close' as 'adj_close' since recent data is already adjusted
        df['adj_close'] = df['close']
        
        # Select and order columns
        columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
        df = df[columns]
        
        # Remove any timezone information from date
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        
        return df
    
    def collect_historical_data_parallel(self, tickers: List[str]) -> None:
        """Collect historical data for multiple tickers in parallel."""
        try:
            parallel_config = self.config.data_collection.parallel_processing
            if not parallel_config.enabled:
                for ticker in tickers:
                    self.collect_historical_data(ticker)
                return
            
            with ThreadPoolExecutor(max_workers=parallel_config.max_workers) as executor:
                executor.map(self.collect_historical_data, tickers)
                
        except Exception as e:
            self.logger.error(f"Error in parallel data collection: {str(e)}")
    
    def start_realtime_collection(self, tickers: List[str]) -> None:
        """Start realtime data collection."""
        if not self.config.data_collection.realtime.enabled:
            self.logger.warning("Realtime data collection is disabled in config")
            return
        
        self._stop_realtime = False
        self.logger.info("Starting realtime data collection")
        
        try:
            while not self._stop_realtime:
                for ticker in tickers:
                    stock = yf.Ticker(ticker)
                    data = stock.history(period="1d", interval="1m")
                    
                    if not data.empty:
                        data = data.reset_index()
                        data.columns = data.columns.str.lower()
                        self.db.save_ticker_data(ticker, data, realtime=True)
                    
                # Wait for next update
                # You might want to adjust this based on your needs
                pd.Timestamp.sleep(timedelta(minutes=1))
                
        except Exception as e:
            self.logger.error(f"Error in realtime data collection: {str(e)}")
        finally:
            self.logger.info("Stopped realtime data collection")
    
    def stop_realtime_collection(self) -> None:
        """Stop realtime data collection."""
        self._stop_realtime = True
    
    def refresh_ticker_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Refresh data for a specific ticker."""
        return self.collect_historical_data(ticker)
    
    def get_ticker_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Get ticker data from the database."""
        try:
            return self.db.get_ticker_data(ticker, start_date, end_date)
        except Exception as e:
            self.logger.error(f"Error getting data for {ticker}: {str(e)}")
            return None 