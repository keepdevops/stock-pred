import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor

class DataLoader:
    """Class for loading stock data from various sources."""
    
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.source = config.get('source', 'yahoo')
        self.start_date = config.get('start_date', '2020-01-01')
        self.end_date = config.get('end_date', 'today')
        
        if self.end_date == 'today':
            self.end_date = datetime.now().strftime('%Y-%m-%d')
            
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def load_data_async(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load stock data asynchronously."""
        try:
            # Run the synchronous load_data method in a thread pool
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(self.executor, self.load_data, symbol)
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data asynchronously: {e}")
            return None
            
    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load stock data from Yahoo Finance."""
        try:
            if self.source == 'yahoo':
                return self._load_from_yahoo(symbol)
            else:
                raise ValueError(f"Unsupported data source: {self.source}")
                
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {e}")
            raise
            
    def _load_from_yahoo(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load stock data from Yahoo Finance."""
        try:
            # Clean the symbol
            symbol = symbol.strip().upper()
            
            # Download data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            
            try:
                data = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval='1d'
                )
            except Exception as e:
                self.logger.error(f"Yahoo Finance API error for {symbol}: {e}")
                raise ValueError(f"Failed to fetch data from Yahoo Finance for {symbol}. Please check if the symbol is valid.")
            
            # Check if data is empty
            if data.empty:
                self.logger.error(f"No data available for symbol {symbol}")
                raise ValueError(f"No data available for symbol {symbol}")
            
            # Reset index to make date a column
            data = data.reset_index()
            
            # Rename columns to match our database schema
            data = data.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            })
            
            # Calculate technical indicators
            data = self.calculate_indicators(data)
            
            self.logger.info(f"Successfully loaded data for {symbol}")
            return data
            
        except ValueError as e:
            raise e
        except Exception as e:
            self.logger.error(f"Error loading data from Yahoo Finance: {e}")
            raise ValueError(f"Failed to load data for {symbol}: {str(e)}")
            
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for the data."""
        try:
            # Moving averages
            data['ma5'] = data['close'].rolling(window=5).mean()
            data['ma20'] = data['close'].rolling(window=20).mean()
            
            # RSI
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            data['rsi'] = 100 - (100 / (1 + rs))
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return data
            
    def get_symbols_list(self) -> list:
        """Get list of symbols from the symbols file."""
        try:
            with open(self.symbols_file, 'r') as f:
                return [line.strip() for line in f if line.strip()]
                
        except Exception as e:
            self.logger.error(f"Error reading symbols file: {e}")
            return []
            
    def update_data(self, symbol: str, db) -> bool:
        """Update stock data for the specified symbol."""
        try:
            # Get the latest date in the database
            latest_data = db.get_stock_data(symbol, limit=1)
            if not latest_data.empty:
                start_date = (pd.to_datetime(latest_data['date'].iloc[0]) + 
                            timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                start_date = self.start_date
                
            # If we're already up to date, no need to update
            if start_date >= self.end_date:
                self.logger.info(f"Data for {symbol} is already up to date")
                return True
                
            # Load new data
            new_data = self.load_data(symbol)
            
            # Filter to only new dates
            new_data = new_data[new_data['date'] > start_date]
            
            if not new_data.empty:
                db.insert_stock_data(symbol, new_data)
                self.logger.info(f"Updated data for {symbol}")
                return True
            else:
                self.logger.info(f"No new data available for {symbol}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating data for {symbol}: {e}")
            return False 

    def __del__(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True) 