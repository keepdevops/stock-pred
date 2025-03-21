"""
Data loader for collecting and processing stock data.
"""
import logging
import yfinance as yf
import pandas as pd
from typing import Optional, Dict
from datetime import datetime

class DataLoader:
    def __init__(self, config):
        """
        Initialize DataLoader.
        
        Args:
            config: DataCollectionConfig instance
        """
        self.logger = logging.getLogger("DataLoader")
        self.config = config

    def collect_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """
        Collect historical data for a given symbol.
        
        Args:
            symbol: Stock symbol
        """
        try:
            self.logger.info(f"Collecting historical data for {symbol}")
            
            # Create Ticker object
            ticker = yf.Ticker(symbol)
            
            # Get data using configuration
            df = ticker.history(
                period=self.config.data_period,
                interval=self.config.data_interval
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

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate processed DataFrame.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            # Check required columns
            missing_cols = set(self.config.required_columns) - set(df.columns)
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Check for NaN values
            if df.isna().any().any():
                self.logger.error("Data contains NaN values")
                return False
            
            # Check data types
            if not (
                pd.api.types.is_datetime64_any_dtype(df['date']) and
                all(pd.api.types.is_numeric_dtype(df[col]) 
                    for col in ['open', 'high', 'low', 'close', 'adj_close', 'volume'])
            ):
                self.logger.error("Invalid data types")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data: {e}")
            return False 