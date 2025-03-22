from typing import List, Optional, Dict, Any
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
import duckdb
import logging
from datetime import datetime
from src.modules.trading.real_trading_agent import RealTradingAgent, TradeConfig
import os
import signal
import psutil
import time

class DatabaseConnector:
    """Handles database connections and queries for stock data."""
    
    def __init__(self, db_path: str, table_name: str = "stock_data", logger=None):
        self.db_path = db_path
        self.table_name = table_name
        self.logger = logger or logging.getLogger(__name__)
        self.conn = None
        self.connect()
        self.recreate_table()

    def _release_db_locks(self):
        """Release any existing locks on the database file."""
        try:
            db_path = Path(self.db_path)
            if not db_path.exists():
                return

            # Check for lock files
            lock_file = Path(str(db_path) + '.lock')
            wal_file = Path(str(db_path) + '.wal')
            
            self.logger.debug(f"Checking for lock files: {lock_file}, {wal_file}")

            # Function to check if a file is locked
            def is_file_locked(filepath):
                for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                    try:
                        for file in proc.open_files():
                            if str(filepath) in file.path:
                                return True, proc.pid, proc.name()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                return False, None, None

            # Check main database file
            is_locked, pid, process_name = is_file_locked(db_path)
            if is_locked:
                self.logger.warning(f"Database locked by process {process_name} (PID: {pid})")
                try:
                    psutil.Process(pid).terminate()
                    time.sleep(1)  # Wait for process to terminate
                    self.logger.info(f"Terminated process {pid}")
                except psutil.NoSuchProcess:
                    pass

            # Remove lock files if they exist
            for file in [lock_file, wal_file]:
                if file.exists():
                    try:
                        file.unlink()
                        self.logger.info(f"Removed lock file: {file}")
                    except Exception as e:
                        self.logger.warning(f"Could not remove lock file {file}: {e}")

            # Wait briefly to ensure files are released
            time.sleep(0.5)

        except Exception as e:
            self.logger.error(f"Error releasing database locks: {e}")
            raise

    def connect(self):
        """Connect to the database with proper lock handling."""
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            # Release any existing locks
            self._release_db_locks()
            
            # Close existing connection if any
            if self.conn is not None:
                try:
                    self.conn.close()
                except:
                    pass
                self.conn = None
            
            # Attempt to connect with retries
            max_retries = 3
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    self.conn = duckdb.connect(
                        database=self.db_path,
                        read_only=False
                    )
                    
                    # Set database pragmas
                    self.conn.execute("SET enable_external_access=true")
                    self.conn.execute("SET enable_object_cache=true")
                    
                    self.logger.info(f"Connected to database: {self.db_path}")
                    break
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                        time.sleep(retry_delay)
                    else:
                        raise
            
        except Exception as e:
            self.logger.error(f"Error connecting to database: {str(e)}")
            raise

    def recreate_table(self):
        """Recreate the stock data table with the correct schema."""
        try:
            # Drop existing table
            self.conn.execute(f"DROP TABLE IF EXISTS {self.table_name}")
            
            # Create new table with all required columns including adj_close
            create_table_sql = f"""
            CREATE TABLE {self.table_name} (
                date TIMESTAMP NOT NULL,
                ticker VARCHAR NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (date, ticker)
            )
            """
            
            self.conn.execute(create_table_sql)
            self.conn.commit()
            
            # Verify the table structure
            self.verify_table_structure()
            
        except Exception as e:
            self.logger.error(f"Error recreating table: {e}")
            raise

    def verify_table_structure(self):
        """Verify the table has the correct structure."""
        try:
            schema = self.conn.execute(f"""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns 
                WHERE table_name = '{self.table_name}'
                ORDER BY ordinal_position
            """).fetchall()
            
            self.logger.info(f"Current {self.table_name} schema:")
            column_names = []
            for col in schema:
                column_names.append(col[0].lower())
                self.logger.info(f"Column: {col[0]}, Type: {col[1]}, Nullable: {col[2]}")
            
            required_columns = [
                'date', 'ticker', 'open', 'high', 'low', 
                'close', 'adj_close', 'volume'
            ]
            
            # Check for missing columns
            missing_columns = set(required_columns) - set(column_names)
            if missing_columns:
                raise ValueError(f"Missing columns: {missing_columns}")
            
            # Check for extra columns
            extra_columns = set(column_names) - set(required_columns)
            if extra_columns:
                raise ValueError(f"Extra columns found: {extra_columns}")
            
            # Verify column count
            if len(column_names) != 8:
                raise ValueError(f"Wrong number of columns: got {len(column_names)}, expected 8")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Table structure verification failed: {e}")
            return False

    def save_ticker_data(self, ticker: str, data: pd.DataFrame) -> None:
        """Save ticker data to database."""
        try:
            self.logger.info(f"Saving data for {ticker}")
            
            # Log incoming data structure
            self.logger.info(f"Incoming data columns: {data.columns.tolist()}")
            self.logger.info("Incoming data types:")
            for col in data.columns:
                self.logger.info(f"  {col}: {data[col].dtype}")
            
            # Ensure data types are correct
            data = data.copy()
            data['date'] = pd.to_datetime(data['date'])
            data['ticker'] = data['ticker'].astype(str)
            numeric_columns = ['open', 'high', 'low', 'close', 'adj_close']
            for col in numeric_columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            data['volume'] = pd.to_numeric(data['volume'], errors='coerce').fillna(0).astype('int64')
            
            # Remove any invalid rows
            data = data.dropna(subset=['date'] + numeric_columns)
            
            # Ensure columns are in correct order
            columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            data = data[columns]
            
            # Delete existing data for this ticker
            self.conn.execute(f"DELETE FROM {self.table_name} WHERE ticker = ?", [ticker])
            
            # Insert new data
            self.conn.execute(f"""
                INSERT INTO {self.table_name} 
                SELECT * FROM read_pandas(?)
            """, [data])
            
            self.conn.commit()
            self.logger.info(f"Successfully saved {len(data)} records for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error saving data for {ticker}: {e}")
            raise

    def get_ticker_data(self, ticker: str) -> pd.DataFrame:
        """Retrieve ticker data from database."""
        try:
            query = f"""
            SELECT date, ticker, open, high, low, close, adj_close, volume
            FROM {self.table_name}
            WHERE ticker = ?
            ORDER BY date
            """
            return self.conn.execute(query, [ticker]).fetchdf()
        except Exception as e:
            self.logger.error(f"Error retrieving data for {ticker}: {str(e)}")
            raise

    def close(self):
        """Properly close the database connection."""
        try:
            if self.conn is not None:
                self.conn.close()
                self.conn = None
                self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")

    def __del__(self):
        """Ensure connection is closed when object is destroyed."""
        self.close()

    def create_connection(self, db_path: Path) -> bool:
        """Create SQLAlchemy engine for DuckDB connection."""
        try:
            self.engine = create_engine(f"duckdb:///{db_path}")
            self.current_db = db_path
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.logger.info(f"Successfully connected to database: {db_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def get_tables(self) -> List[str]:
        """Get all tables in current database."""
        if not self.engine:
            self.logger.warning("No active database connection")
            return []
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main'"
                ))
                return [row[0] for row in result]
        except Exception as e:
            self.logger.error(f"Error getting tables: {e}")
            return []
    
    def load_tickers(self, table: str) -> pd.DataFrame:
        """Load ticker data from specified table."""
        if not self.engine:
            self.logger.warning("No active database connection")
            return pd.DataFrame()
        
        try:
            query = f"""
            SELECT date, open, high, low, close, adj_close, volume, ticker, adj_close
            FROM {table}
            ORDER BY date, ticker
            """
            return pd.read_sql(query, self.engine)
            
        except Exception as e:
            self.logger.error(f"Error loading tickers from {table}: {e}")
            return pd.DataFrame()
    
    def get_unique_tickers(self, table: str) -> List[str]:
        """Get list of unique tickers in a table."""
        if not self.engine:
            return []
        
        try:
            query = f"SELECT DISTINCT ticker FROM {table} ORDER BY ticker"
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                return [row[0] for row in result]
        except Exception as e:
            self.logger.error(f"Error getting unique tickers: {e}")
            return []
    
    def save_training_results(
        self,
        model_name: str,
        ticker: str,
        metrics: Dict[str, float],
        predictions: pd.DataFrame
    ) -> bool:
        """Save training results to training_data.duckdb."""
        try:
            training_db = self.current_db.parent / "training_data.duckdb"
            train_engine = create_engine(f"duckdb:///{training_db}")
            
            # Save metrics
            metrics_df = pd.DataFrame([{
                'model_name': model_name,
                'ticker': ticker,
                'timestamp': datetime.now(),
                **metrics
            }])
            
            metrics_df.to_sql(
                'training_metrics',
                train_engine,
                if_exists='append',
                index=False
            )
            
            # Save predictions
            predictions['model_name'] = model_name
            predictions['ticker'] = ticker
            predictions['timestamp'] = datetime.now()
            
            predictions.to_sql(
                'predictions',
                train_engine,
                if_exists='append',
                index=False
            )
            
            self.logger.info(f"Saved training results for {ticker} using {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving training results: {e}")
            return False
    
    def create_sector_tables(self, sector_mappings: Dict[str, List[str]]) -> bool:
        """Create tables for different market sectors."""
        if not self.engine:
            return False
        
        try:
            with self.engine.connect() as conn:
                # Create sector tables if they don't exist
                for sector, tickers in sector_mappings.items():
                    table_name = f"{sector.lower()}_stocks"
                    
                    # Create table
                    conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        date DATETIME,
                        open FLOAT,
                        high FLOAT,
                        low FLOAT,
                        close FLOAT,
                        adj_close FLOAT,
                        volume FLOAT,
                        ticker VARCHAR,
                        dDATETIME
                    )
                    """))
                    
                    # Create indexes
                    conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_date_ticker 
                    ON {table_name}(date, ticker)
                    """))
                    
                    self.logger.info(f"Created sector table: {table_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating sector tables: {e}")
            return False
    
    def update_stock_data(
        self,
        df: pd.DataFrame,
        table: str,
        if_exists: str = 'append'
    ) -> bool:
        """Update stock data in specified table."""
        if not self.engine:
            return False
        
        try:
            # Add updated_at timestamp
            df['updated_at'] = datetime.now()
            
            # Write to database
            df.to_sql(
                table,
                self.engine,
                if_exists=if_exists,
                index=False,
                method='multi',
                chunksize=1000
            )
            
            self.logger.info(f"Updated {len(df)} rows in {table}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating stock data: {e}")
            return False
    
    def get_latest_data(
        self,
        table: str,
        ticker: str,
        days: int = 30
    ) -> pd.DataFrame:
        """Get latest N days of data for a ticker."""
        if not self.engine:
            return pd.DataFrame()
        
        try:
            query = f"""
            SELECT *
            FROM {table}
            WHERE ticker = :ticker
            ORDER BY date DESC
            LIMIT :days
            """
            
            return pd.read_sql(
                query,
                self.engine,
                params={'ticker': ticker, 'days': days}
            )
            
        except Exception as e:
            self.logger.error(f"Error getting latest data: {e}")
            return pd.DataFrame()
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> bool:
        """Remove data older than specified days."""
        if not self.engine:
            return False
        
        try:
            with self.engine.connect() as conn:
                # Get all tables
                tables = self.get_tables()
                
                for table in tables:
                    conn.execute(text(f"""
                    DELETE FROM {table}
                    WHERE date < DATEADD('day', -{days_to_keep}, CURRENT_DATE)
                    """))
                    
                self.logger.info(f"Cleaned up old data (keeping {days_to_keep} days)")
                return True
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            return False

    def initialize_tables(self):
        """Initialize database tables with correct schema."""
        try:
            # Drop existing table to ensure clean slate
            self.conn.execute("DROP TABLE IF EXISTS stock_data")
            
            # Create table with all required columns including adj_close
            create_table_sql = """
            CREATE TABLE stock_data (
                date TIMESTAMP NOT NULL,
                ticker VARCHAR NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (date, ticker)
            )
            """
            
            self.conn.execute(create_table_sql)
            self.conn.commit()
            
            # Verify the table structure
            self.conn.execute("""
                SELECT column_name, data_type, is_nullable, is_primary_key
                FROM information_schema.columns 
                WHERE table_name = 'stock_data'
                ORDER BY ordinal_position
            """)
            columns = self.conn.execute("""
                SELECT column_name, data_type, is_nullable, is_primary_key
                FROM information_schema.columns 
                WHERE table_name = 'stock_data'
                ORDER BY ordinal_position
            """).fetchall()
            self.logger.info("Table structure:")
            for col in columns:
                self.logger.info(f"  {col}")
            
            self.logger.info("Database tables initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise 