import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import glob
import json
import polars as pl
from pathlib import Path
import traceback

class DataLoader:
    """Class for loading and managing stock market data."""
    
    def __init__(self, db_connector):
        """Initialize the DataLoader with a database connector."""
        self.logger = logging.getLogger(__name__)
        self.db_connector = db_connector
        
        # Default configuration
        self.source = 'yahoo'
        self.start_date = '2020-01-01'
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        
        # Initialize thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize cache
        self.data_cache = {}
        self.last_update = {}
        
        self.logger.info("DataLoader initialized successfully")
    
    def __del__(self):
        """Clean up resources."""
        try:
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
        except Exception as e:
            self.logger.error(f"Error closing DataLoader: {str(e)}")
    
    async def load_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load stock data for the given symbol."""
        try:
            # Use provided dates or defaults
            start = start_date or self.start_date
            end = end_date or self.end_date
            
            # Check cache first
            cache_key = f"{symbol}_{start}_{end}"
            if cache_key in self.data_cache:
                last_update = self.last_update.get(cache_key)
                if last_update and (datetime.now() - last_update).seconds < 300:  # 5 minutes cache
                    return self.data_cache[cache_key]
            
            # Load data from database first
            data = self.db_connector.get_stock_data(symbol, start, end)
            
            # If no data in database or data is old, fetch from API
            if data is None or data.empty:
                data = await self.fetch_from_api(symbol, start, end)
                if not data.empty:
                    self.db_connector.save_stock_data(symbol, data)
            
            # Update cache
            self.data_cache[cache_key] = data
            self.last_update[cache_key] = datetime.now()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    async def load_from_database(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load data from the database."""
        try:
            # Use the correct database connector method
            data = await self.db_connector.get_stock_data(symbol, start_date, end_date)
            if data is not None and not data.empty:
                self.logger.info(f"Successfully loaded data from database for {symbol}")
                return data
            self.logger.info(f"No data found in database for {symbol}")
            return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error loading data from database for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    async def fetch_from_api(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from the API."""
        try:
            # Use yfinance for now, can be extended to support other sources
            import yfinance as yf
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                self.executor,
                lambda: yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)
            )
            
            if data.empty:
                self.logger.warning(f"No data available from API for {symbol}")
                return pd.DataFrame()
                
            # Clean and format data
            data = data.reset_index()
            
            # Print column names for debugging
            self.logger.info(f"API data columns before processing: {data.columns.tolist()}")
            
            # Rename columns to match our schema
            column_mapping = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            }
            
            # Only map columns that exist in the data
            existing_columns = {k: v for k, v in column_mapping.items() if k in data.columns}
            data = data.rename(columns=existing_columns)
            
            # Print column names after renaming
            self.logger.info(f"API data columns after renaming: {data.columns.tolist()}")
            
            # If adj_close is missing, use close as adj_close
            if 'adj_close' not in data.columns and 'close' in data.columns:
                data['adj_close'] = data['close']
                self.logger.info("Using close price as adj_close since Adj Close is not available")
            
            # Ensure required columns exist
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
            if not all(col in data.columns for col in required_columns):
                missing = [col for col in required_columns if col not in data.columns]
                self.logger.error(f"Missing required columns in API data: {missing}")
                return pd.DataFrame()
            
            # Convert date column to datetime if needed
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
            
            # Print final column names and data shape
            self.logger.info(f"Final data columns: {data.columns.tolist()}")
            self.logger.info(f"Data shape: {data.shape}")
            
            self.logger.info(f"Successfully fetched data from API for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data from API for {symbol}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    async def save_to_database(self, symbol: str, data: pd.DataFrame):
        """Save data to the database."""
        try:
            # Save data using the database connector's save_stock_data method
            await self.db_connector.save_stock_data(symbol, data)
            self.logger.info(f"Successfully saved data to database for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error saving data to database for {symbol}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def load_data_async(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load stock data asynchronously."""
        try:
            self.logger.info(f"Loading data for {symbol}")
            
            # Load data from database first
            data = self.db_connector.get_stock_data(symbol, self.start_date, self.end_date)
            
            # If no data in database or data is old, fetch from API
            if data is None or data.empty:
                self.logger.info(f"No data in database for {symbol}, fetching from API")
                data = await self.fetch_from_api(symbol, self.start_date, self.end_date)
                if not data.empty:
                    self.logger.info(f"Successfully fetched data from API for {symbol}")
                    self.db_connector.save_stock_data(symbol, data)
                else:
                    self.logger.warning(f"No data available from API for {symbol}")
                    return pd.DataFrame()
            
            if data.empty:
                self.logger.warning(f"No data available for {symbol}")
                return pd.DataFrame()
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
            
    def load_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load stock data synchronously."""
        try:
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run the async method
            data = loop.run_until_complete(self.load_data_async(symbol))
            
            # Close the loop
            loop.close()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error in load_data for {symbol}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
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

    def close(self):
        """Clean up resources."""
        try:
            self.executor.shutdown(wait=True)
            self.logger.info("DataLoader resources cleaned up")
        except Exception as e:
            self.logger.error(f"Error closing DataLoader: {e}")
            
    def fetch_stock_data(self, symbol: str, start_date: Optional[datetime] = None, 
                        end_date: Optional[datetime] = None) -> Tuple[pd.DataFrame, str]:
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for data fetch
            end_date: End date for data fetch
            
        Returns:
            Tuple of (DataFrame with stock data, error message if any)
        """
        try:
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=365)
                
            # Fetch data from Yahoo Finance
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                return None, f"No data found for symbol {symbol}"
                
            # Ensure required columns exist
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                return None, f"Missing required columns: {', '.join(missing)}"
                
            # Rename columns to lowercase
            df.columns = df.columns.str.lower()
            
            # Add date index as a column
            df['date'] = df.index
            
            self.logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df, None
            
        except Exception as e:
            error_msg = f"Error fetching data for {symbol}: {str(e)}"
            self.logger.error(error_msg)
            return None, error_msg
            
    def validate_symbol(self, symbol: str) -> Tuple[bool, str]:
        """
        Validate if a stock symbol exists.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            Tuple of (is_valid, error message if any)
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            if not info:
                return False, f"Symbol {symbol} not found"
                
            return True, None
            
        except Exception as e:
            return False, f"Error validating symbol {symbol}: {str(e)}"
            
    def get_company_info(self, symbol: str) -> Tuple[dict, str]:
        """
        Get company information from Yahoo Finance.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Tuple of (company info dictionary, error message if any)
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            
            if not info:
                return None, f"No information found for symbol {symbol}"
                
            # Extract relevant information
            company_info = {
                'name': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('forwardPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0)
            }
            
            return company_info, None
            
        except Exception as e:
            return None, f"Error fetching company info for {symbol}: {str(e)}"

    def _load_from_directory(self, symbol: str) -> Optional[pd.DataFrame]:
        """Load stock data from directory containing various file formats."""
        try:
            symbol = symbol.strip().upper()
            data_dir = Path(self.data_dir)
            
            # Try each file type
            for file_type in self.file_types:
                pattern = f"**/*{symbol}*.{file_type}"
                files = list(data_dir.glob(pattern))
                
                if files:
                    # Use the most recent file if multiple exist
                    file_path = max(files, key=lambda x: x.stat().st_mtime)
                    
                    try:
                        if file_type == 'csv':
                            df = pd.read_csv(file_path)
                        elif file_type == 'json':
                            df = pd.read_json(file_path)
                        elif file_type == 'parquet':
                            df = pd.read_parquet(file_path)
                        elif file_type == 'pkl':
                            df = pd.read_pickle(file_path)
                        else:
                            continue
                            
                        # Ensure required columns exist
                        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                        if not all(col in df.columns for col in required_columns):
                            missing = [col for col in required_columns if col not in df.columns]
                            self.logger.warning(f"Missing required columns in {file_path}: {missing}")
                            continue
                            
                        # Convert date column to datetime if needed
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                            
                        # Calculate technical indicators
                        df = self.calculate_indicators(df)
                        
                        self.logger.info(f"Successfully loaded data for {symbol} from {file_path}")
                        return df
                        
                    except Exception as e:
                        self.logger.error(f"Error loading file {file_path}: {e}")
                        continue
                        
            raise ValueError(f"No valid data files found for symbol {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error loading data from directory: {e}")
            raise
            
    def load_from_polars(self, df: pl.DataFrame) -> pd.DataFrame:
        """
        Convert Polars DataFrame to Pandas DataFrame with required format.
        
        Args:
            df: Polars DataFrame containing stock data
            
        Returns:
            Pandas DataFrame with required format
        """
        try:
            # Convert to pandas
            pandas_df = df.to_pandas()
            
            # Ensure required columns exist
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in pandas_df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in pandas_df.columns]
                raise ValueError(f"Missing required columns: {', '.join(missing)}")
                
            # Convert date column to datetime if needed
            if 'date' in pandas_df.columns:
                pandas_df['date'] = pd.to_datetime(pandas_df['date'])
                
            # Calculate technical indicators
            pandas_df = self.calculate_indicators(pandas_df)
            
            return pandas_df
            
        except Exception as e:
            self.logger.error(f"Error converting Polars DataFrame: {e}")
            raise
            
    def load_from_sqlite(self, db_path: str, query: str) -> pd.DataFrame:
        """
        Load stock data from SQLite database.
        
        Args:
            db_path: Path to SQLite database file
            query: SQL query to fetch data
            
        Returns:
            Pandas DataFrame with stock data
        """
        try:
            import sqlite3
            
            # Connect to database
            conn = sqlite3.connect(db_path)
            
            # Execute query and load into DataFrame
            df = pd.read_sql_query(query, conn)
            
            # Close connection
            conn.close()
            
            # Ensure required columns exist
            required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in df.columns]
                raise ValueError(f"Missing required columns: {', '.join(missing)}")
                
            # Convert date column to datetime if needed
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                
            # Calculate technical indicators
            df = self.calculate_indicators(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from SQLite: {e}")
            raise
            
    def list_available_symbols(self) -> List[str]:
        """List all available stock symbols in the data directory."""
        try:
            symbols = set()
            data_dir = Path(self.data_dir)
            
            # Find all files matching the supported file types
            for file_type in self.file_types:
                for file_path in data_dir.glob(f"**/*.{file_type}"):
                    if file_path.is_file():
                        # Extract symbol from filename (assuming format: SYMBOL_*.ext)
                        symbol = file_path.stem.split('_')[0].upper()
                        symbols.add(symbol)
                        
            return sorted(list(symbols))
            
        except Exception as e:
            self.logger.error(f"Error listing available symbols: {e}")
            return []
            
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about a data file."""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            info = {
                'filename': file_path.name,
                'extension': file_path.suffix.lower(),
                'size': file_path.stat().st_size,
                'modified': datetime.fromtimestamp(file_path.stat().st_mtime),
                'symbol': file_path.stem.split('_')[0].upper()
            }
            
            # Try to get data preview
            try:
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path, nrows=5)
                elif file_path.suffix.lower() == '.json':
                    df = pd.read_json(file_path)
                elif file_path.suffix.lower() == '.parquet':
                    df = pd.read_parquet(file_path)
                elif file_path.suffix.lower() == '.pkl':
                    df = pd.read_pickle(file_path)
                else:
                    df = None
                    
                if df is not None:
                    info['columns'] = list(df.columns)
                    info['rows'] = len(df)
                    info['date_range'] = {
                        'start': df['date'].min() if 'date' in df.columns else None,
                        'end': df['date'].max() if 'date' in df.columns else None
                    }
                    
            except Exception as e:
                self.logger.warning(f"Could not get data preview: {e}")
                
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting file info: {e}")
            raise 