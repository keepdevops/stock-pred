from typing import Optional
from datetime import datetime
import pandas as pd
import logging
import duckdb
import polars as pl
from pathlib import Path

class DatabaseManager:
    def __init__(self, db_path: str = "stocks.db"):
        self.db_path = db_path
        self.connection = None
        self.initialize_database()

    def initialize_database(self):
        """Initialize the database connection and create tables"""
        try:
            self.connection = duckdb.connect(self.db_path)
            self.drop_and_create_tables()
            logging.info("Database initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing database: {str(e)}")
            raise

    def drop_and_create_tables(self):
        """Drop existing tables and create new ones"""
        try:
            # Drop existing table if it exists
            self.connection.execute("""
                DROP TABLE IF EXISTS stock_data
            """)
            
            # Create new table with updated schema
            self.connection.execute("""
                CREATE TABLE stock_data (
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
            
            # Create indexes for better query performance
            self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_date ON stock_data(date)
            """)
            self.connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_stock_ticker ON stock_data(ticker)
            """)
            
            self.connection.commit()
            logging.info("Database tables recreated successfully")
        except Exception as e:
            logging.error(f"Error recreating tables: {str(e)}")
            raise

    def close(self):
        """Close the database connection"""
        if self.connection:
            self.connection.close()

    def get_last_update(self, ticker: str) -> Optional[datetime]:
        """
        Get the last update date for a ticker
        
        Args:
            ticker (str): The ticker symbol to check
            
        Returns:
            Optional[datetime]: The last update date or None if not found
        """
        try:
            query = """
                SELECT MAX(date) as last_update
                FROM stock_data
                WHERE ticker = ?
            """
            result = self.connection.execute(query, (ticker,)).fetchone()
            return result[0] if result and result[0] else None
        except Exception as e:
            logging.error(f"Error getting last update for {ticker}: {str(e)}")
            return None

    def get_ticker_data(self, ticker: str) -> pd.DataFrame:
        """
        Get all data for a specific ticker
        
        Args:
            ticker (str): The ticker symbol
        Returns:
            pd.DataFrame: DataFrame containing ticker data
        """
        try:
            query = """
                SELECT date, ticker, open, high, low, close, adj_close, volume
                FROM stock_data
                WHERE ticker = ?
                ORDER BY date
            """
            result = self.connection.execute(query, (ticker,)).fetch_arrow_table()
            return pl.from_arrow(result).to_pandas()
        except Exception as e:
            logging.error(f"Error getting data for {ticker}: {str(e)}")
            return pd.DataFrame()

    def get_latest_prices(self, ticker: str) -> Optional[dict]:
        """
        Get the most recent price data for a ticker
        
        Args:
            ticker (str): The ticker symbol
        Returns:
            Optional[dict]: Dictionary containing latest price data
        """
        try:
            query = """
                SELECT date, open, high, low, close, adj_close, volume
                FROM stock_data
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 1
            """
            result = self.connection.execute(query, (ticker,)).fetchone()
            if result:
                return {
                    'date': result[0],
                    'open': result[1],
                    'high': result[2],
                    'low': result[3],
                    'close': result[4],
                    'adj_close': result[5],
                    'volume': result[6]
                }
            return None
        except Exception as e:
            logging.error(f"Error getting latest prices for {ticker}: {str(e)}")
            return None

    def insert_stock_data(self, df: pd.DataFrame) -> bool:
        """
        Insert stock data into database
        """
        if df is None or df.empty:
            return False
            
        try:
            # Ensure column names are lowercase
            df.columns = df.columns.str.lower()
            
            required_columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            
            # Verify all required columns exist
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Select only the required columns in the correct order
            df_to_insert = df[required_columns].copy()
            
            # Insert data
            self.connection.execute("""
                INSERT INTO stock_data 
                (date, ticker, open, high, low, close, adj_close, volume)
                SELECT * FROM df_to_insert
            """)
            self.connection.commit()
            return True
            
        except Exception as e:
            logging.error(f"Error inserting data: {str(e)}")
            return False

    def create_tables(self):
        """
        Create the necessary database tables
        """
        try:
            self.connection.execute("""
                DROP TABLE IF EXISTS stock_data;
                
                CREATE TABLE stock_data (
                    date DATE,
                    ticker VARCHAR,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    adj_close DOUBLE,
                    volume DOUBLE,
                    PRIMARY KEY (date, ticker)
                );
                
                CREATE INDEX IF NOT EXISTS idx_stock_date ON stock_data(date);
                CREATE INDEX IF NOT EXISTS idx_stock_ticker ON stock_data(ticker);
            """)
            self.connection.commit()
            logging.info("Database tables created successfully")
        except Exception as e:
            logging.error(f"Error creating tables: {str(e)}")
            raise

    def verify_table_structure(self):
        """
        Verify the table structure and available columns
        """
        try:
            # Get table information
            table_info = self.connection.execute("""
                DESCRIBE stock_data
            """).fetchall()
            
            # Log available columns
            columns = [row[0] for row in table_info]
            logging.info(f"Available columns in stock_data: {columns}")
            
            # Verify required columns exist
            required_columns = ['date', 'ticker', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            missing_columns = [col for col in required_columns if col not in columns]
            
            if missing_columns:
                logging.error(f"Missing required columns: {missing_columns}")
                return False
                
            return True
        except Exception as e:
            logging.error(f"Error verifying table structure: {str(e)}")
            return False

    def verify_data(self, ticker: str):
        """
        Verify data for a specific ticker
        """
        try:
            result = self.connection.execute("""
                SELECT 
                    COUNT(*) as count,
                    MIN(date) as first_date,
                    MAX(date) as last_date
                FROM stock_data
                WHERE ticker = ?
            """, (ticker,)).fetchone()
            
            logging.info(f"Ticker {ticker}: {result[0]} records from {result[1]} to {result[2]}")
            return True
        except Exception as e:
            logging.error(f"Error verifying data for {ticker}: {str(e)}")
            return False

    def get_all_symbols(self):
        """Get all symbols from the database"""
        try:
            # Try to read from the most recent ticker file
            data_dir = Path("data")
            ticker_files = sorted(
                data_dir.glob("tickers_*.csv"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if ticker_files:
                # Read using polars for better performance
                df = pl.read_csv(ticker_files[0]).to_pandas()
                return df
            else:
                logging.warning("No ticker files found")
                return pd.DataFrame()
                
        except Exception as e:
            logging.error(f"Error reading symbols: {e}")
            return pd.DataFrame() 