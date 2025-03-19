import duckdb
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List, Dict

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