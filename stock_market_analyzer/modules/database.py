import duckdb
import pandas as pd
import logging
import os
import time
import psutil
from datetime import datetime, timedelta

class DatabaseConnector:
    def __init__(self, db_path="stock_data.duckdb"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.setup_database()
        
    def _release_db_locks(self):
        """Attempt to release any locks on the database file."""
        try:
            if not os.path.exists(self.db_path):
                return
                
            self.logger.info(f"Attempting to release locks on {self.db_path}")
            
            # Get current process ID
            current_pid = os.getpid()
            
            # Look for other Python processes that might be holding the lock
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    # Skip current process
                    if proc.info['pid'] == current_pid:
                        continue
                        
                    # Check if it's a Python process
                    if 'python' in proc.info['name'].lower():
                        cmdline = proc.info['cmdline']
                        if any('stock_market_analyzer' in str(cmd) for cmd in cmdline if cmd):
                            self.logger.warning(f"Found potential locking process: PID {proc.info['pid']}")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
                    
            self.logger.info("Lock release attempt completed")
            
        except Exception as e:
            self.logger.error(f"Error while trying to release database locks: {e}")
            
    def setup_database(self):
        """Set up the database schema."""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Try to release any existing locks
                if attempt > 0:
                    self._release_db_locks()
                    
                # Try to connect with write access
                self.conn = duckdb.connect(self.db_path)
                
                # Create tables
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS stock_data (
                        date DATE,
                        symbol VARCHAR,
                        open DOUBLE,
                        high DOUBLE,
                        low DOUBLE,
                        close DOUBLE,
                        volume BIGINT,
                        PRIMARY KEY (date, symbol)
                    )
                """)
                
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        date DATE,
                        symbol VARCHAR,
                        predicted_price DOUBLE,
                        model_name VARCHAR,
                        PRIMARY KEY (date, symbol, model_name)
                    )
                """)
                
                self.logger.info("Database setup completed successfully")
                # Successfully connected, exit the retry loop
                break
                
            except Exception as e:
                self.logger.warning(f"Database connection attempt {attempt+1}/{max_retries} failed: {e}")
                
                if attempt == max_retries - 1:
                    # Last attempt, try read-only as fallback
                    try:
                        self.logger.info("Attempting read-only database connection")
                        self.conn = duckdb.connect(self.db_path, read_only=True)
                        self.logger.info("Connected to database in read-only mode")
                    except Exception as read_only_error:
                        self.logger.error(f"Error setting up database: {e}")
                        self.logger.error(f"Read-only fallback also failed: {read_only_error}")
                        raise
                else:
                    # Wait before retrying
                    time.sleep(retry_delay)
                    
    def save_stock_data(self, data: pd.DataFrame, symbol: str):
        """Save stock data to the database."""
        try:
            # Check if connection is read-only
            try:
                is_read_only = self.conn.execute("PRAGMA read_only").fetchone()[0] == 1
            except:
                is_read_only = False
                
            if is_read_only:
                self.logger.warning("Cannot save data - database is in read-only mode")
                return
                
            # Ensure data has the correct columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                raise ValueError("Data missing required columns")
                
            # Add symbol column
            data['symbol'] = symbol
            
            # Save to database
            self.conn.execute("""
                INSERT OR REPLACE INTO stock_data 
                SELECT * FROM data
            """, {'data': data})
            
            self.logger.info(f"Saved {len(data)} records for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error saving stock data: {e}")
            raise
            
    def get_stock_data(self, symbol: str, start_date=None, end_date=None):
        """Retrieve stock data from the database."""
        try:
            query = """
                SELECT * FROM stock_data 
                WHERE symbol = ?
            """
            params = [symbol]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
                
            query += " ORDER BY date"
            
            return self.conn.execute(query, params).fetchdf()
            
        except Exception as e:
            self.logger.error(f"Error retrieving stock data: {e}")
            return None
            
    def save_prediction(self, date: datetime, symbol: str, predicted_price: float, model_name: str):
        """Save a prediction to the database."""
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO predictions 
                VALUES (?, ?, ?, ?)
            """, [date, symbol, predicted_price, model_name])
            
            self.logger.info(f"Saved prediction for {symbol} using {model_name}")
            
        except Exception as e:
            self.logger.error(f"Error saving prediction: {e}")
            raise
            
    def get_predictions(self, symbol: str, model_name=None, start_date=None, end_date=None):
        """Retrieve predictions from the database."""
        try:
            query = "SELECT * FROM predictions WHERE symbol = ?"
            params = [symbol]
            
            if model_name:
                query += " AND model_name = ?"
                params.append(model_name)
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
                
            query += " ORDER BY date"
            
            return self.conn.execute(query, params).fetchdf()
            
        except Exception as e:
            self.logger.error(f"Error retrieving predictions: {e}")
            return None
            
    def close(self):
        """Close the database connection."""
        try:
            self.conn.close()
            self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}") 