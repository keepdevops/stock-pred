"""
NASDAQ symbols database handler.
"""
import logging
import duckdb
import pandas as pd
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import glob
import os

class NasdaqDatabase:
    def __init__(self):
        self.logger = logging.getLogger("NasdaqDB")
        self.symbols = []
        self.symbols_data = None
        self.load_nasdaq_symbols()

    def load_nasdaq_symbols(self):
        """Load symbols from NASDAQ screener CSV file."""
        try:
            self.logger.info("Loading NASDAQ symbols...")
            
            # Look for the NASDAQ screener file in the project directory
            nasdaq_files = glob.glob("nasdaq_screener_*.csv")
            if not nasdaq_files:
                self.logger.error("No NASDAQ screener CSV file found")
                return False
            
            file_path = nasdaq_files[0]
            self.logger.info(f"Found NASDAQ file: {file_path}")
            
            # Read the CSV file
            self.symbols_data = pd.read_csv(file_path)
            
            # Extract and sort symbols
            self.symbols = sorted(self.symbols_data['Symbol'].unique().tolist())
            
            self.logger.info(f"Successfully loaded {len(self.symbols)} NASDAQ symbols")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading NASDAQ symbols: {e}")
            return False

    def get_all_symbols(self):
        """Return all loaded symbols."""
        return self.symbols

    def get_symbol_info(self, symbol):
        """Get detailed information for a specific symbol."""
        if self.symbols_data is not None and symbol in self.symbols:
            info = self.symbols_data[self.symbols_data['Symbol'] == symbol]
            if not info.empty:
                return info.iloc[0].to_dict()
        return None

    def search_symbols(self, search_text):
        """Search symbols based on text."""
        search_text = search_text.upper()
        return [symbol for symbol in self.symbols if search_text in symbol]

    def _initialize_connection(self) -> duckdb.DuckDBPyConnection:
        """Initialize database connection and create tables."""
        try:
            # Create directory if it doesn't exist
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Connect to database
            conn = duckdb.connect(str(self.db_path))

            # Create tables if they don't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nasdaq_symbols (
                    symbol VARCHAR PRIMARY KEY,
                    name VARCHAR,
                    sector VARCHAR,
                    industry VARCHAR,
                    market_cap DOUBLE,
                         date TIMESTAMP
                )
            """)

            self.logger.info(f"Connected to NASDAQ database: {self.db_path}")
            return conn

        except Exception as e:
            self.logger.error(f"Error initializing NASDAQ database: {e}")
            raise

    def import_nasdaq_screener(self, csv_path: Path) -> bool:
        """
        Import NASDAQ screener CSV file.
        
        Args:
            csv_path: Path to NASDAQ screener CSV
        """
        try:
            # Read CSV file
            df = pd.read_csv(csv_path)

            # Clean and standardize data
            df = self._clean_nasdaq_data(df)

            # Begin transaction
            self.conn.begin()

            try:
                # Clear existing data
                self.conn.execute("DELETE FROM nasdaq_symbols")

                # Insert new data
                self.conn.execute("""
                    INSERT INTO nasdaq_symbols 
                    SELECT * FROM df
                """)

                # Commit transaction
                self.conn.commit()

                self.logger.info(f"Imported {len(df)} NASDAQ symbols")
                return True

            except Exception as e:
                self.conn.rollback()
                raise e

        except Exception as e:
            self.logger.error(f"Error importing NASDAQ symbols: {e}")
            return False

    def _clean_nasdaq_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize NASDAQ data."""
        try:
            # Rename columns
            df = df.rename(columns={
                'Symbol': 'symbol',
                'Name': 'name',
                'Sector': 'sector',
                'Industry': 'industry',
                'Market Cap': 'market_cap'
            })

            # Clean symbol names
            df['symbol'] = df['symbol'].apply(self._clean_symbol)

            # Add last updated timestamp
            df['last_updated'] = datetime.now()

            return df

        except Exception as e:
            self.logger.error(f"Error cleaning NASDAQ data: {e}")
            raise

    def _clean_symbol(self, symbol: str) -> str:
        """Clean symbol string."""
        if not isinstance(symbol, str):
            return str(symbol)

        # Convert to uppercase and strip whitespace
        cleaned = symbol.strip().upper()

        # Replace special characters
        replacements = {
            '^': '-P-',    # Preferred shares
            '/': '-W-',    # Warrants
            '=': '-U-',    # Units
            '$': '-D-',    # Debentures
            '.': '-',      # Class shares
            ' ': '-',      # Spaces
            '+': '-PLUS-', # Special cases
        }

        for char, replacement in replacements.items():
            cleaned = cleaned.replace(char, replacement)

        # Remove duplicate dashes
        while '--' in cleaned:
            cleaned = cleaned.replace('--', '-')

        return cleaned.strip('-')

    def close(self):
        """Close database connection."""
        try:
            if self.conn:
                self.conn.close()
                self.logger.info("NASDAQ database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing NASDAQ database: {e}") 