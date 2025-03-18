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
    
    def __init__(
        self,
        db_connector: DatabaseConnector,
        config: ConfigurationManager,
        logger: Optional[logging.Logger] = None
    ):
        self.db = db_connector
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self._stop_realtime = False
    
    def collect_historical_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Collect historical data for a given ticker."""
        try:
            self.logger.info(f"Collecting historical data for {ticker}")
            
            # Get configuration
            hist_config = self.config.data_collection.historical
            
            # Download data
            stock = yf.Ticker(ticker)
            df = stock.history(
                start=hist_config.start_date,
                end=hist_config.end_date
            )
            
            if df.empty:
                self.logger.warning(f"No historical data found for {ticker}")
                return None
            
            # Process the data
            df = df.reset_index()
            df.columns = df.columns.str.lower()
            
            # Save to database
            self.db.save_ticker_data(ticker, df)
            
            self.logger.info(f"Successfully collected historical data for {ticker}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting historical data for {ticker}: {str(e)}")
            return None
    
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