import logging
from pathlib import Path
from typing import Optional, Dict, List
import duckdb
import pandas as pd
import os
import signal
import psutil

class DatabaseConnector:
    """Handles database connections and operations."""
    
    def __init__(self, db_path=None, logger=None):
        self.logger = logger or logging.getLogger("modules.database")
        
        # Handle both string and dict inputs for backward compatibility
        if isinstance(db_path, dict):
            self.db_path = db_path.get('path', 'data/market_data.duckdb')
        elif isinstance(db_path, str):
            self.db_path = db_path
        else:
            self.db_path = 'data/market_data.duckdb'
            
        self.connection = None
        self.connect(force=True)  # Add force parameter
    
    def connect(self, force=False):
        """Establish database connection."""
        try:
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(exist_ok=True)
            
            if force:
                self._release_db_locks()
            
            self.connection = duckdb.connect(str(self.db_path))
            self.initialize_tables()
            self.logger.info(f"Connected to database: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Error connecting to database: {e}")
            raise
    
    def _release_db_locks(self):
        """Release any existing locks on the database."""
        try:
            db_path = Path(self.db_path).resolve()
            
            # Find processes that might have the database open
            current_pid = os.getpid()
            
            for proc in psutil.process_iter(['pid', 'name', 'open_files']):
                try:
                    if proc.pid == current_pid:
                        continue
                        
                    # Check if process has the database file open
                    for file in proc.open_files() or []:
                        if Path(file.path).resolve() == db_path:
                            self.logger.info(f"Found process holding database lock: {proc.pid}")
                            # Try to terminate the process
                            os.kill(proc.pid, signal.SIGTERM)
                            self.logger.info(f"Released lock from process: {proc.pid}")
                            
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
                    
            # Additional cleanup: remove lock file if it exists
            lock_file = Path(str(self.db_path) + ".lock")
            if lock_file.exists():
                lock_file.unlink()
                self.logger.info("Removed database lock file")
                
        except Exception as e:
            self.logger.warning(f"Error releasing database locks: {e}")
    
    def initialize_tables(self):
        """Initialize database tables."""
        try:
            # Create stock_data table if it doesn't exist
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    date DATE NOT NULL,
                    ticker VARCHAR NOT NULL,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    adj_close DOUBLE,
                    volume BIGINT,
                    PRIMARY KEY (date, ticker)
                )
            """)
            
            # Log table structure
            table_info = self.connection.execute("DESCRIBE stock_data").fetchall()
            self.logger.info("Database tables initialized")
            self.logger.info(f"Table structure: {table_info}")
            
        except Exception as e:
            self.logger.error(f"Error initializing tables: {e}")
            raise
    
    def execute_query(self, query, params=None):
        """Execute a query and return results as DataFrame."""
        try:
            if params:
                result = self.connection.execute(query, params).fetchdf()
            else:
                result = self.connection.execute(query).fetchdf()
            return result
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            raise
    
    def close(self):
        """Close database connection."""
        if self.connection:
            try:
                self.connection.close()
                self.logger.info("Database connection closed")
            except Exception as e:
                self.logger.error(f"Error closing database connection: {e}")
    
    def __del__(self):
        """Ensure connection is closed when object is deleted."""
        self.close()
    
    def save_ticker_data(
        self,
        ticker: str,
        data: pd.DataFrame,
        realtime: bool = False
    ) -> None:
        """Save ticker data to database."""
        try:
            # Prepare data
            df = data.copy()
            df['ticker'] = ticker
            
            # Insert data
            if realtime:
                # For realtime data, we might want to update existing records
                self.connection.execute("""
                    INSERT OR REPLACE INTO stock_data
                    SELECT * FROM df
                """)
            else:
                # For historical data, ignore duplicates
                self.connection.execute("""
                    INSERT OR IGNORE INTO stock_data
                    SELECT * FROM df
                """)
            
            self.logger.info(f"Saved data for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error saving data for {ticker}: {str(e)}")
            raise
    
    def save_trade(
        self,
        ticker: str,
        action: str,
        quantity: float,
        price: float
    ) -> None:
        """Save trade to history."""
        try:
            total_value = quantity * price
            
            self.connection.execute("""
                INSERT INTO trading_history (
                    trade_id, date, ticker, action, quantity, price, total_value
                )
                SELECT 
                    nextval('trading_history_trade_id_seq'),
                    CURRENT_TIMESTAMP,
                    ?, ?, ?, ?, ?
            """, [ticker, action, quantity, price, total_value])
            
            self.logger.info(f"Saved trade for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error saving trade: {str(e)}")
            raise
    
    def get_ticker_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Retrieve ticker data from database."""
        try:
            query = "SELECT * FROM stock_data WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            result = self.connection.execute(query, params).fetchdf()
            return result if not result.empty else None
            
        except Exception as e:
            self.logger.error(f"Error retrieving data for {ticker}: {str(e)}")
            return None
    
    def get_trading_history(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Retrieve trading history."""
        try:
            query = "SELECT * FROM trading_history"
            params = []
            
            if ticker:
                query += " WHERE ticker = ?"
                params.append(ticker)
            
            if start_date:
                query += " AND date >= ?" if ticker else " WHERE date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date DESC"
            
            return self.connection.execute(query, params).fetchdf()
            
        except Exception as e:
            self.logger.error(f"Error retrieving trading history: {str(e)}")
            return pd.DataFrame()
    
    def get_tables(self) -> List[str]:
        """Get list of tables in the database."""
        try:
            tables = self.connection.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
            """).fetchdf()
            return tables['table_name'].tolist()
        except Exception:
            # Try SQLite syntax if DuckDB fails
            try:
                tables = self.connection.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table'
                """).fetchdf()
                return tables['name'].tolist()
            except Exception as e:
                self.logger.error(f"Error getting tables: {str(e)}")
                return [] 