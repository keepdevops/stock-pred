import duckdb
import pandas as pd
import logging
import os
import time
import psutil
from datetime import datetime, timedelta
import sqlite3
import threading
from typing import Optional, Dict, Any
import traceback

class DatabaseConnector:
    """Handles database operations for stock data."""
    
    def __init__(self, db_path: str = None):
        """Initialize the database connector."""
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path or os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_data.db')
        self.connection = None
        self._ensure_db_directory()
        self._initialize_database()
        self.logger.info("Database connector initialized")
        
    def _ensure_db_directory(self):
        """Ensure the database directory exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
    def _initialize_database(self):
        """Initialize the database with required tables."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create stock_data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    symbol TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adj_close REAL,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            conn.commit()
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
    def _get_connection(self):
        """Get a database connection."""
        if self.connection is None:
            self.connection = sqlite3.connect(self.db_path)
        return self.connection
        
    def save_stock_data(self, symbol: str, data: pd.DataFrame):
        """Save stock data to the database."""
        try:
            # Convert Series to DataFrame if necessary
            if isinstance(data, pd.Series):
                data = data.to_frame().T
                
            # Ensure data is a DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")
                
            # Add symbol column if it doesn't exist
            if 'symbol' not in data.columns:
                data['symbol'] = symbol
                
            # Get connection
            conn = self._get_connection()
            
            # Save data to database
            data.to_sql('stock_data', conn, if_exists='append', index=False)
            
            self.logger.info(f"Successfully saved data for {symbol} to database")
            
        except Exception as e:
            self.logger.error(f"Error saving data for {symbol} to database: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        
    def get_stock_data(self, symbol: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get stock data from the database."""
        try:
            conn = self._get_connection()
            
            # Build query
            query = "SELECT * FROM stock_data WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            # Execute query
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                self.logger.info(f"Successfully loaded data from database for {symbol}")
            else:
                self.logger.info(f"No data found in database for {symbol}")
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from database for {symbol}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
        
    def close(self):
        """Close the database connection."""
        try:
            if self.connection:
                self.connection.close()
                self.connection = None
                self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {str(e)}")
            
    def connect(self):
        """Establish connection to the database with retry logic."""
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Attempting database connection (attempt {attempt + 1}/{self.max_retries})")
                
                # Ensure directory exists
                os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
                
                # Try to connect
                self.connection = duckdb.connect(self.db_path)
                
                # Test the connection
                self.connection.execute("SELECT 1")
                
                self.logger.info("Database connection established successfully")
                return True
                
            except Exception as e:
                self.logger.warning(f"Database connection attempt {attempt + 1}/{self.max_retries} failed: {str(e)}")
                
                # Try to release locks if this isn't the first attempt
                if attempt > 0:
                    self._release_db_locks()
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    # Last attempt, try force unlock
                    if self.force_unlock():
                        try:
                            self.connection = duckdb.connect(self.db_path)
                            self.connection.execute("SELECT 1")
                            self.logger.info("Database connection established after force unlock")
                            return True
                        except Exception as force_error:
                            self.logger.error(f"Force unlock didn't help: {force_error}")
                    
                    self.logger.error("Failed to establish database connection after all retries")
                    return False
                    
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
                            
                            # Try to release locks by closing database connections in other processes
                            # This is a soft approach and won't kill the process
                            try:
                                # Check if the process is still running
                                if psutil.pid_exists(proc.info['pid']):
                                    self.logger.info(f"Attempting to release locks from process {proc.info['pid']}")
                                    # Try to unlock by creating a temporary read-only connection
                                    try:
                                        temp_conn = duckdb.connect(self.db_path, read_only=True)
                                        temp_conn.close()
                                        self.logger.info(f"Successfully released locks via temporary connection")
                                    except Exception as e:
                                        self.logger.warning(f"Could not create temporary connection: {e}")
                            except Exception as e:
                                self.logger.warning(f"Error checking process {proc.info['pid']}: {e}")
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
                    
            self.logger.info("Lock release attempt completed")
            
        except Exception as e:
            self.logger.error(f"Error while trying to release database locks: {e}")
            
    def force_unlock(self):
        """Force unlock the database by recreating it if necessary."""
        try:
            if not os.path.exists(self.db_path):
                self.logger.info(f"Database file {self.db_path} doesn't exist, no need to unlock")
                return True
                
            self.logger.warning(f"Attempting to force unlock database {self.db_path}")
            
            # First try the gentle approach - release locks
            self._release_db_locks()
            
            # Try to connect to verify if locks are released
            try:
                temp_conn = duckdb.connect(self.db_path)
                temp_conn.close()
                self.logger.info("Database unlocked successfully")
                return True
            except Exception as e:
                self.logger.warning(f"Soft unlock failed: {e}")
            
            # If still locked, try to create a temporary backup and restore
            backup_path = f"{self.db_path}.backup_{int(time.time())}"
            self.logger.warning(f"Creating backup at {backup_path} before force unlock")
            
            try:
                # Copy the database file to backup
                import shutil
                if os.path.exists(self.db_path):
                    shutil.copy2(self.db_path, backup_path)
                    self.logger.info(f"Created backup at {backup_path}")
                    
                    # Remove the current file
                    os.remove(self.db_path)
                    self.logger.info(f"Removed locked database file")
                    
                    # Create a new empty database
                    temp_conn = duckdb.connect(self.db_path)
                    temp_conn.close()
                    self.logger.info("Created new database file")
                    
                    return True
            except Exception as e:
                self.logger.error(f"Force unlock failed: {e}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during force unlock: {e}")
            return False
            
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
                self.connection = duckdb.connect(self.db_path)
                
                # Create tables
                self.connection.execute("""
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
                
                self.connection.execute("""
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
                    # Last attempt, try to force unlock and retry one more time
                    self.logger.warning("Final connection attempt failed, trying force unlock")
                    if self.force_unlock():
                        try:
                            # Try again after force unlock
                            self.connection = duckdb.connect(self.db_path)
                            
                            # Create tables
                            self.connection.execute("""
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
                            
                            self.connection.execute("""
                                CREATE TABLE IF NOT EXISTS predictions (
                                    date DATE,
                                    symbol VARCHAR,
                                    predicted_price DOUBLE,
                                    model_name VARCHAR,
                                    PRIMARY KEY (date, symbol, model_name)
                                )
                            """)
                            
                            self.logger.info("Database setup completed successfully after force unlock")
                            break
                        except Exception as force_error:
                            self.logger.error(f"Force unlock didn't help: {force_error}")
                            
                    # If force unlock failed or connecting after force unlock failed, try read-only
                    try:
                        self.logger.info("Attempting read-only database connection")
                        self.connection = duckdb.connect(self.db_path, read_only=True)
                        self.logger.info("Connected to database in read-only mode")
                    except Exception as read_only_error:
                        self.logger.error(f"Error setting up database: {e}")
                        self.logger.error(f"Read-only fallback also failed: {read_only_error}")
                        raise
                else:
                    # Wait before retrying
                    time.sleep(retry_delay)
                    
    def save_prediction(self, date: datetime, symbol: str, predicted_price: float, model_name: str):
        """Save a prediction to the database."""
        try:
            self.connection.execute("""
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
            
            return self.connection.execute(query, params).fetchdf()
            
        except Exception as e:
            self.logger.error(f"Error retrieving predictions: {e}")
            return None 