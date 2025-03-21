import duckdb
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import time
import psutil
import os
from datetime import datetime, date
from threading import Lock
import requests
from dataclasses import dataclass
import threading

@dataclass
class LoadingProgress:
    total_records: int = 0
    processed_records: int = 0
    current_operation: str = ""
    status: str = "idle"
    error: Optional[str] = None
    retry_count: int = 0
    start_time: float = 0

class DatabaseManager:
    def __init__(self, db_path: str = "stocks.db", logger=None):
        """
        Initialize DatabaseManager
        
        Args:
            db_path (str): Path to the database file
            logger: Logger instance
        """
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        self.lock = Lock()  # Add thread lock
        self.connection = None
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        self.loading_cancelled = False
        self.progress = LoadingProgress()
        self.initialize_database()

    def _release_db_locks(self):
        """Attempt to release any existing database locks."""
        try:
            db_path = str(self.db_path.absolute())
            for proc in psutil.process_iter(['pid', 'name', 'username']):
                try:
                    # Check if process has the database file open
                    for item in proc.open_files():
                        if db_path in item.path:
                            self.logger.warning(f"Found lock by process {proc.pid}")
                            if proc.pid != os.getpid():
                                proc.terminate()
                                self.logger.info(f"Terminated process {proc.pid}")
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            self.logger.error(f"Error releasing locks: {e}")

    def initialize_database(self):
        """Initialize the database and create necessary tables"""
        try:
            with self.lock:
                self.connection = duckdb.connect(self.db_path)
                
                # Create tables
                self.create_tables()
                
                # Check if we need to load symbols
                count = self.get_symbols_count()
                if count == 0:
                    self.loading_cancelled = False
                    self.load_initial_data()

                self.logger.info(f"Database initialized with {self.get_symbols_count()} symbols")
                
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            raise

    def create_tables(self):
        """Create necessary database tables"""
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                date DATE,
                ticker VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume DOUBLE,
                PRIMARY KEY (date, ticker)
            )
        """)

        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS stock_symbols (
                symbol VARCHAR PRIMARY KEY,
                name VARCHAR,
                exchange VARCHAR,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

    def load_initial_data(self):
        """Load initial data from CSV file"""
        # Try different paths for the data file
        possible_paths = [
            "data/all_stocks_1742447810.csv",
            "./data/all_stocks_1742447810.csv",
            "../data/all_stocks_1742447810.csv",
            os.path.join(os.path.dirname(__file__), "../../data/all_stocks_1742447810.csv")
        ]

        for path in possible_paths:
            if self.loading_cancelled:
                self.logger.info("Loading cancelled by user")
                return False
                
            if os.path.exists(path):
                self.logger.info(f"Loading stock data from: {path}")
                if self.load_symbols_from_csv(path):
                    return True

        self.logger.warning("No stock data file found, loading defaults")
        return self.load_default_symbols()

    def verify_table_structure(self):
        """Verify the table structure and available columns"""
        try:
            # Get table information
            table_info = self.connection.execute("""
                DESCRIBE stock_data
            """).fetchall()
            
            # Log available columns
            columns = [row[0] for row in table_info]
            self.logger.info(f"Available columns in stock_data: {columns}")
            
            # Verify required columns exist
            required_columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            missing_columns = [col for col in required_columns if col not in columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
                
            return True
        except Exception as e:
            self.logger.error(f"Error verifying table structure: {str(e)}")
            return False

    def __enter__(self):
        """Context manager enter"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def close(self):
        """Close the database connection"""
        try:
            with self.lock:  # Add thread safety
                if self.connection:
                    self.connection.close()
                    self.connection = None
        except Exception as e:
            self.logger.error(f"Error closing database: {str(e)}")

    def insert_data(self, df: pd.DataFrame):
        """Insert stock data with retry mechanism."""
        for attempt in range(self.max_retries):
            try:
                # Add updated_at column with current timestamp
                df['updated_at'] = pd.Timestamp.now()
                
                # Ensure required columns exist
                required_columns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'adj']
                for col in required_columns:
                    if col not in df.columns:
                        raise ValueError(f"Required column {col} not found in data")
                
                # Prepare data for insertion
                insert_df = df.rename(columns={
                    'Date': 'date',
                    'Symbol': 'ticker',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume'
                })
                
                # Insert data with UPSERT logic
                self.connection.execute("""
                    INSERT OR REPLACE INTO stock_data 
                    SELECT 
                        date,
                        ticker,
                        open,
                        high,
                        low,
                        close,
                        volume,
                        
                    FROM insert_df
                """)
                
                self.logger.info(f"Inserted {len(df)} rows for {df['Symbol'].iloc[0]} at {pd.Timestamp.now()}")
                return
                
            except duckdb.IOException as e:
                if "Conflicting lock" in str(e) and attempt < self.max_retries - 1:
                    self.logger.warning(f"Database locked during insert, attempt {attempt + 1} of {self.max_retries}")
                    time.sleep(self.retry_delay)
                    self.initialize_database()  # Reconnect
                    continue
                raise
            except Exception as e:
                self.logger.error(f"Error inserting data: {e}")
                raise

    def get_last_update(self, ticker: str) -> Optional[date]:
        """Get the last update date for a ticker"""
        try:
            # Validate ticker before query
            if not ticker or pd.isna(ticker) or str(ticker).lower() == 'nan':
                return None
                
            with self.lock:
                query = """
                    SELECT MAX(date) as last_update
                    FROM stock_data
                    WHERE ticker = ?
                """
                result = self.connection.execute(query, (str(ticker),)).fetchone()
                
                if result and result[0]:
                    # Convert the result to a date object
                    if isinstance(result[0], datetime):
                        return result[0].date()
                    elif isinstance(result[0], str):
                        return datetime.strptime(result[0], '%Y-%m-%d').date()
                    elif isinstance(result[0], date):
                        return result[0]
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting last update for {ticker}: {str(e)}")
            return None

    def insert_stock_data(self, df: pd.DataFrame) -> bool:
        """Insert stock data into database"""
        if df is None or df.empty:
            return False
            
        try:
            with self.lock:
                # Handle MultiIndex if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                # Ensure column names are lowercase
                df.columns = df.columns.str.lower()
                
                # Convert date column to datetime
                df['date'] = pd.to_datetime(df['date']).dt.date
                
                # Verify required columns
                required_columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}")
                
                # Select only required columns in correct order
                df_to_insert = df[required_columns].copy()
                
                # Insert data
                self.connection.execute("""
                    INSERT OR REPLACE INTO stock_data 
                    (date, ticker, open, high, low, close, adj_close, volume)
                    SELECT * FROM df_to_insert
                """)
                self.connection.commit()
                return True
                
        except Exception as e:
            self.logger.error(f"Error inserting data: {str(e)}")
            return False

    def get_symbols_count(self) -> int:
        """Get count of symbols in database"""
        try:
            result = self.connection.execute("SELECT COUNT(*) FROM stock_symbols").fetchone()
            return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"Error getting symbols count: {str(e)}")
            return 0

    def get_all_symbols(self) -> pd.DataFrame:
        """Get all symbols as DataFrame"""
        try:
            query = """
                SELECT 
                    symbol,
                    name,
                    exchange,
                    last_updated
                FROM stock_symbols
                ORDER BY symbol
            """
            return self.connection.execute(query).fetchdf()
        except Exception as e:
            self.logger.error(f"Error getting symbols: {str(e)}")
            return pd.DataFrame()

    def load_symbols_from_csv(self, file_path: str) -> bool:
        """Load symbols from CSV file with progress tracking and validation"""
        try:
            self.progress = LoadingProgress(start_time=time.time())
            self.progress.current_operation = "Reading file"
            self.progress.status = "running"

            # First pass: count records and validate structure
            with pd.read_csv(file_path, chunksize=1000) as reader:
                self.progress.total_records = sum(len(chunk) for chunk in reader)

            # Validate file structure
            df_sample = pd.read_csv(file_path, nrows=1)
            required_columns = {'Symbol', 'Name', 'Exchange'}
            if not all(col in df_sample.columns for col in required_columns):
                raise ValueError(f"Missing required columns. Found: {df_sample.columns.tolist()}")

            # Read and process in chunks
            chunk_size = 1000
            chunks = pd.read_csv(file_path, chunksize=chunk_size)
            
            with self.lock:
                self.connection.execute("BEGIN TRANSACTION")
                self.connection.execute("DELETE FROM stock_symbols")
                
                for chunk in chunks:
                    if self.loading_cancelled:
                        self.connection.execute("ROLLBACK")
                        self.progress.status = "cancelled"
                        return False

                    # Clean and validate chunk data
                    chunk = self._clean_and_validate_chunk(chunk)
                    
                    # Insert valid records
                    if not chunk.empty:
                        self.connection.execute("""
                            INSERT INTO stock_symbols (symbol, name, exchange)
                            SELECT symbol, name, exchange FROM chunk
                        """)
                    
                    self.progress.processed_records += len(chunk)
                    self.progress.current_operation = f"Processing records {self.progress.processed_records}/{self.progress.total_records}"

                self.connection.execute("COMMIT")
                self.progress.status = "completed"
                return True

        except Exception as e:
            self.logger.error(f"Error loading symbols: {str(e)}")
            self.progress.error = str(e)
            self.progress.status = "error"
            return self._handle_loading_error()

    def _clean_and_validate_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate data chunk"""
        try:
            # Clean data
            chunk = chunk.assign(
                symbol=chunk['Symbol'].str.strip(),
                name=chunk['Name'].str.strip(),
                exchange=chunk['Exchange'].str.strip()
            )

            # Validate data
            valid_mask = (
                chunk['symbol'].notna() &
                chunk['symbol'].str.len().between(1, 10) &
                chunk['name'].notna() &
                chunk['exchange'].notna()
            )

            # Log invalid records
            invalid_records = chunk[~valid_mask]
            if not invalid_records.empty:
                self.logger.warning(f"Found {len(invalid_records)} invalid records")
                for _, row in invalid_records.iterrows():
                    self.logger.debug(f"Invalid record: {row.to_dict()}")

            return chunk[valid_mask].drop_duplicates(subset=['symbol'])

        except Exception as e:
            self.logger.error(f"Error cleaning chunk: {str(e)}")
            raise

    def _handle_loading_error(self) -> bool:
        """Handle loading errors with retry logic"""
        if self.progress.retry_count < self.max_retries:
            self.progress.retry_count += 1
            self.logger.info(f"Retrying operation (attempt {self.progress.retry_count}/{self.max_retries})")
            time.sleep(self.retry_delay)
            return self.load_symbols_from_csv(self.current_file)
        return False

    def get_loading_status(self) -> Dict:
        """Get current loading status"""
        if self.progress.total_records > 0:
            percentage = (self.progress.processed_records / self.progress.total_records) * 100
            elapsed_time = time.time() - self.progress.start_time
            records_per_second = self.progress.processed_records / elapsed_time if elapsed_time > 0 else 0
            
            return {
                'status': self.progress.status,
                'progress': percentage,
                'current_operation': self.progress.current_operation,
                'processed_records': self.progress.processed_records,
                'total_records': self.progress.total_records,
                'error': self.progress.error,
                'retry_count': self.progress.retry_count,
                'elapsed_time': elapsed_time,
                'records_per_second': records_per_second
            }
        return {'status': self.progress.status}

    def load_default_symbols(self):
        """Load default symbols"""
        try:
            default_symbols = [
                ('AAPL', 'Apple Inc.', 'NASDAQ'),
                ('MSFT', 'Microsoft Corporation', 'NASDAQ'),
                ('GOOGL', 'Alphabet Inc.', 'NASDAQ'),
                ('AMZN', 'Amazon.com Inc.', 'NASDAQ'),
                ('META', 'Meta Platforms Inc.', 'NASDAQ')
            ]
            
            self.connection.execute("DELETE FROM stock_symbols")
            self.connection.execute("""
                INSERT INTO stock_symbols (symbol, name, exchange)
                VALUES (?, ?, ?)
            """, default_symbols)
            self.connection.commit()
            
            self.logger.info("Loaded default symbols")
            return True
        except Exception as e:
            self.logger.error(f"Error loading default symbols: {str(e)}")
            return False

    def update_nasdaq_data(self):
        """Update NASDAQ data from official source"""
        try:
            self.logger.info("Updating NASDAQ data...")
            
            # NASDAQ URL
            nasdaq_url = "https://api.nasdaq.com/api/screener/stocks"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Download data
            response = requests.get(nasdaq_url, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            if 'data' not in data or 'rows' not in data['data']:
                raise ValueError("Invalid data format from NASDAQ API")
            
            # Convert to DataFrame
            df = pd.DataFrame(data['data']['rows'])
            
            # Clean and prepare data
            df = df.assign(
                symbol=df['symbol'].str.strip(),
                name=df['name'].str.strip(),
                exchange=df['exchange'].str.strip()
            ).dropna(subset=['symbol'])
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['symbol'])
            
            # Save to database
            with self.lock:
                self.connection.execute("DELETE FROM stock_symbols")
                self.connection.execute("""
                    INSERT INTO stock_symbols (symbol, name, exchange)
                    SELECT symbol, name, exchange FROM df
                """)
                self.connection.commit()
            
            count = self.get_symbols_count()
            self.logger.info(f"Successfully updated {count} symbols from NASDAQ")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating NASDAQ data: {str(e)}")
            return False

    def load_nasdaq_tickers(self):
        """Load NASDAQ tickers from database or update if needed"""
        try:
            count = self.get_symbols_count()
            if count == 0:
                self.logger.info("No tickers found in database")
                return self.update_nasdaq_data()
            
            self.logger.info(f"Loaded {count} tickers from database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading NASDAQ tickers: {str(e)}")
            return False

    def cancel_loading(self):
        """Cancel ongoing data loading"""
        self.loading_cancelled = True 