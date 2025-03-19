import duckdb
import pandas as pd
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import yfinance as yf

class DataCollector:
    def __init__(self, db_path="data/market_data.duckdb"):
        """Initialize the database connection."""
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.conn = None
        self.connect()
        self.initialize_tables()

    def __del__(self):
        """Cleanup database connection."""
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()

    def connect(self):
        """Establish database connection."""
        try:
            self.logger.info(f"Connected to database: {self.db_path}")
            self.conn = duckdb.connect(self.db_path)
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise

    def initialize_tables(self):
        """Initialize database tables."""
        try:
            # Create tables if they don't exist
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    date DATE,
                    ticker VARCHAR(20) NOT NULL,  -- Increased length for cleaned symbols
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    adj_close DOUBLE,
                    volume BIGINT,
                    PRIMARY KEY (date, ticker)
                )
            """)
            
            self.logger.info("Database tables initialized successfully")
            
            # Log the table structure
            result = self.conn.execute("""
                SELECT column_name, data_type, is_nullable, column_default, 
                       numeric_precision, numeric_scale
                FROM information_schema.columns 
                WHERE table_name = 'stock_data'
                ORDER BY ordinal_position
            """).fetchall()
            
            self.logger.info(f"Table structure: {result}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {str(e)}")
            return False

    def download_yfinance_data(self, ticker, start_date, end_date):
        """Download stock data from Yahoo Finance."""
        try:
            # Download data
            self.logger.info(f"Downloading data for {ticker} from {start_date} to {end_date}")
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Rename columns to match database schema
            df.columns = [col.lower() for col in df.columns]
            df = df.rename(columns={'stock splits': 'splits'})
            
            # Add ticker column
            df['ticker'] = ticker
            
            # Ensure correct column order
            df = df[['date', 'ticker', 'open', 'high', 'low', 'close', 'adj close', 'volume']]
            
            # Save to database
            self.save_ticker_data(df)
            
            self.logger.info(f"Successfully downloaded and saved data for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Failed to download data for {ticker}: {e}")
            raise

    def save_ticker_data(self, df):
        """Save ticker data to database."""
        try:
            if df.empty:
                self.logger.warning("Empty DataFrame received, skipping save")
                return False
            
            # Log incoming data
            self.logger.info(f"Saving data - Shape: {df.shape}")
            
            # Get ticker
            ticker = df['ticker'].iloc[0]
            
            try:
                # Start transaction
                self.conn.execute("BEGIN TRANSACTION")
                
                # Delete existing data
                self.conn.execute("""
                    DELETE FROM stock_data 
                    WHERE ticker = ?
                """, [ticker])
                
                # Insert new data
                self.conn.execute("""
                    INSERT INTO stock_data (
                        date, ticker, open, high, low, close, adj_close, volume
                    ) SELECT 
                        date, ticker, open, high, low, close, adj_close, volume 
                    FROM df
                """)
                
                # Commit transaction
                self.conn.execute("COMMIT")
                
                # Verify the save
                count = self.conn.execute("""
                    SELECT COUNT(*) FROM stock_data WHERE ticker = ?
                """, [ticker]).fetchone()[0]
                
                self.logger.info(f"Saved {count} rows for {ticker}")
                return True
                
            except Exception as e:
                self.logger.error(f"Database error for {ticker}: {str(e)}")
                self.conn.execute("ROLLBACK")
                return False
            
        except Exception as e:
            self.logger.error(f"Failed to save ticker data: {str(e)}")
            return False

    def import_csv_file(self, file_path):
        """Import data from CSV file."""
        try:
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Process and save data
            self.save_ticker_data(df)
            
            self.logger.info(f"Successfully imported data from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to import CSV file: {e}")
            raise

    def get_ticker_data(self, ticker, start_date=None, end_date=None):
        """Retrieve ticker data from database."""
        try:
            query = """
                SELECT * FROM stock_data 
                WHERE ticker = ?
            """
            params = [ticker]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
                
            query += " ORDER BY date"
            
            return self.conn.execute(query, params).fetchdf()
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve data for {ticker}: {e}")
            raise

class DatabaseConnector:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.connect()
        self.initialize_tables()  # Initialize tables on creation

    def connect(self):
        """Connect to the database"""
        try:
            # Ensure data directory exists
            Path("data").mkdir(exist_ok=True)
            
            self.connection = duckdb.connect("data/market_data.duckdb")
            self.cursor = self.connection.cursor()
            self.logger.info("Connected to database: data/market_data.duckdb")
            
        except Exception as e:
            self.logger.error(f"Error connecting to database: {e}")
            raise

    def initialize_tables(self):
        """Initialize database tables"""
        try:
            # Create stock data table with correct schema
            self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                date DATE,
                ticker VARCHAR(20) NOT NULL,  -- Increased length for cleaned symbols
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (date, ticker)
            )
            """)
            self.connection.commit()
        except Exception as e:
            self.logger.error(f"Error initializing tables: {e}")
            raise

    def verify_table_structure(self):
        """Verify that the table has the correct structure."""
        try:
            self.cursor.execute("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'stock_data'
                ORDER BY ordinal_position
            """)
            columns = [col[0] for col in self.cursor.fetchall()]
            expected_columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            
            if columns != expected_columns:
                self.logger.warning(f"Table structure mismatch. Found: {columns}")
                self.logger.info("Recreating table with correct structure...")
                self.initialize_tables()
                return False
            return True
        except Exception as e:
            self.logger.error(f"Error verifying table structure: {e}")
            raise

    def save_ticker_data(self, ticker: str, data: pd.DataFrame, realtime: bool = False) -> None:
        """Save ticker data to database."""
        try:
            # If realtime, only delete overlapping time periods
            if realtime:
                min_date = data['date'].min()
                max_date = data['date'].max()
                self.cursor.execute("""
                    DELETE FROM stock_data 
                    WHERE ticker = ? 
                    AND date BETWEEN ? AND ?
                """, (ticker, min_date, max_date))
            else:
                # For historical data, delete all existing data for the ticker
                self.cursor.execute("DELETE FROM stock_data WHERE ticker = ?", (ticker,))
            
            # Insert new data
            self.cursor.executemany(
                """
                INSERT INTO stock_data (date, ticker, open, high, low, close, adj_close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                data.values.tolist()
            )
            
            self.connection.commit()
            
        except Exception as e:
            self.logger.error(f"Error saving data for {ticker}: {str(e)}")
            raise

    def close(self):
        """Close database connection"""
        try:
            if hasattr(self, 'connection') and self.connection:
                self.connection.close()
                self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")

    def __del__(self):
        """Destructor to ensure connection is closed"""
        self.close() 