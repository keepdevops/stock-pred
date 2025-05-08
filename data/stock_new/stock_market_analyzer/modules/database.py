import sqlite3
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import json

class DatabaseConnector:
    """Database connector for stock market data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the database connector.
        
        Args:
            config: Database configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.db_path = self.config.get('name', 'stock_data.db')
        self.conn = None
        self.setup_database()
        
    def setup_database(self) -> None:
        """Set up the database and create necessary tables."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            cursor = self.conn.cursor()
            
            # Create stock data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stock_data (
                    symbol TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adj_close REAL,
                    PRIMARY KEY (symbol, date)
                )
            ''')
            
            # Create technical indicators table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    symbol TEXT,
                    date TEXT,
                    ma5 REAL,
                    ma20 REAL,
                    rsi REAL,
                    macd REAL,
                    macd_signal REAL,
                    macd_hist REAL,
                    bollinger_upper REAL,
                    bollinger_middle REAL,
                    bollinger_lower REAL,
                    PRIMARY KEY (symbol, date)
                )
            ''')
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    symbol TEXT,
                    date TEXT,
                    predicted_price REAL,
                    confidence REAL,
                    model_name TEXT,
                    features_used TEXT,
                    PRIMARY KEY (symbol, date, model_name)
                )
            ''')
            
            self.conn.commit()
            self.logger.info("Database setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error setting up database: {e}")
            raise
            
    def _ensure_connection(self) -> None:
        """Ensure database connection is established."""
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_path)
            
    def _close_connection(self) -> None:
        """Close database connection if open."""
        if self.conn is not None:
            self.conn.close()
            self.conn = None
                
    def insert_stock_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Insert stock data into the database.
        
        Args:
            symbol: Stock symbol
            data: DataFrame containing stock data
        """
        try:
            self._ensure_connection()
            
            # Prepare data for insertion
            data['symbol'] = symbol
            data['date'] = data['date'].astype(str)
            
            # Insert data
            data.to_sql('stock_data', self.conn, if_exists='append', index=False)
            self.logger.info(f"Inserted {len(data)} records for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error inserting stock data: {e}")
            raise
        finally:
            self._close_connection()
                
    def get_stock_data(self, symbol: str, start_date: Optional[str] = None,
                      end_date: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Get stock data from the database.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            limit: Maximum number of records to retrieve
            
        Returns:
            DataFrame containing stock data
        """
        try:
            self._ensure_connection()
            
            # Build query
            query = f"SELECT * FROM stock_data WHERE symbol = '{symbol}'"
            if start_date:
                query += f" AND date >= '{start_date}'"
            if end_date:
                query += f" AND date <= '{end_date}'"
            query += " ORDER BY date"
            if limit:
                query += f" LIMIT {limit}"
                
            # Execute query
            data = pd.read_sql_query(query, self.conn)
            
            # Convert date column to datetime
            if not data.empty:
                data['date'] = pd.to_datetime(data['date'])
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error retrieving stock data: {e}")
            raise
        finally:
            self._close_connection()
                
    def insert_technical_indicators(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Insert technical indicators into the database.
        
        Args:
            symbol: Stock symbol
            data: DataFrame containing technical indicators
        """
        try:
            self._ensure_connection()
            
            # Prepare data for insertion
            data['symbol'] = symbol
            data['date'] = data['date'].astype(str)
            
            # Insert data
            data.to_sql('technical_indicators', self.conn, if_exists='append', index=False)
            self.logger.info(f"Inserted technical indicators for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error inserting technical indicators: {e}")
            raise
        finally:
            self._close_connection()
                
    def get_technical_indicators(self, symbol: str, start_date: Optional[str] = None,
                              end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get technical indicators from the database.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame containing technical indicators
        """
        try:
            self._ensure_connection()
            
            # Build query
            query = f"SELECT * FROM technical_indicators WHERE symbol = '{symbol}'"
            if start_date:
                query += f" AND date >= '{start_date}'"
            if end_date:
                query += f" AND date <= '{end_date}'"
            query += " ORDER BY date"
            
            # Execute query
            data = pd.read_sql_query(query, self.conn)
            
            # Convert date column to datetime
            if not data.empty:
                data['date'] = pd.to_datetime(data['date'])
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error retrieving technical indicators: {e}")
            raise
        finally:
            self._close_connection()
                
    def insert_predictions(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Insert predictions into the database.
        
        Args:
            symbol: Stock symbol
            data: DataFrame containing predictions
        """
        try:
            self._ensure_connection()
            
            # Prepare data for insertion
            data['symbol'] = symbol
            data['date'] = data['date'].astype(str)
            
            # Insert data
            data.to_sql('predictions', self.conn, if_exists='append', index=False)
            self.logger.info(f"Inserted predictions for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error inserting predictions: {e}")
            raise
        finally:
            self._close_connection()
                
    def get_predictions(self, symbol: str, start_date: Optional[str] = None,
                       end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get predictions from the database.
        
        Args:
            symbol: Stock symbol
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            
        Returns:
            DataFrame containing predictions
        """
        try:
            self._ensure_connection()
            
            # Build query
            query = f"SELECT * FROM predictions WHERE symbol = '{symbol}'"
            if start_date:
                query += f" AND date >= '{start_date}'"
            if end_date:
                query += f" AND date <= '{end_date}'"
            query += " ORDER BY date"
            
            # Execute query
            data = pd.read_sql_query(query, self.conn)
            
            # Convert date column to datetime
            if not data.empty:
                data['date'] = pd.to_datetime(data['date'])
                
            return data
            
        except Exception as e:
            self.logger.error(f"Error retrieving predictions: {e}")
            raise
        finally:
            self._close_connection()
                
    def close(self) -> None:
        """Close the database connection."""
        self._close_connection()
        self.logger.info("Database connection closed") 