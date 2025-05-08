import sqlite3
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import psutil
from datetime import datetime

class DatabaseConnector:
    """Class to handle database connections and operations."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.db_path = self.project_root / "data" / "stock_data.db"
        self.logger = logging.getLogger(__name__)
        
        # Ensure the data directory exists
        os.makedirs(self.db_path.parent, exist_ok=True)
        
        # Initialize connection
        self.conn = None
        self.cursor = None
        self.connect()
        
    def connect(self):
        """Establish a connection to the SQLite database."""
        try:
            self.conn = sqlite3.connect(str(self.db_path))
            self.cursor = self.conn.cursor()
            self.logger.info(f"Connected to database at {self.db_path}")
            
            # Initialize tables if they don't exist
            self._initialize_database()
            
        except sqlite3.Error as e:
            self.logger.error(f"Error connecting to database: {e}")
            raise
    
    def disconnect(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")
    
    def get_stock_data(self, symbol: Optional[str] = None, market_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve stock data from the database.
        
        Args:
            symbol: Optional filter by stock symbol
            market_type: Optional filter by market type
        
        Returns:
            List of dictionaries containing stock data
        """
        try:
            conditions = []
            params = []
            
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            
            if market_type:
                conditions.append("market_type = ?")
                params.append(market_type)
            
            query = "SELECT * FROM stock_data"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            self.cursor.execute(query, params)
            columns = [description[0] for description in self.cursor.description]
            results = []
            
            for row in self.cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
            
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving stock data: {e}")
            return []
    
    def insert_stock_data(self, data: Dict[str, Any]) -> bool:
        """
        Insert stock data into the database.
        
        Args:
            data: Dictionary containing stock data
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            query = f"INSERT INTO stock_data ({columns}) VALUES ({placeholders})"
            
            self.cursor.execute(query, list(data.values()))
            self.conn.commit()
            
            self.logger.info(f"Inserted stock data for symbol: {data.get('symbol')}")
            return True
            
        except sqlite3.Error as e:
            self.logger.error(f"Error inserting stock data: {e}")
            self.conn.rollback()
            return False
    
    def update_stock_data(self, symbol: str, data: Dict[str, Any]) -> bool:
        """
        Update stock data in the database.
        
        Args:
            symbol: Stock symbol to update
            data: Dictionary containing updated stock data
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            set_clause = ", ".join([f"{k} = ?" for k in data.keys()])
            query = f"UPDATE stock_data SET {set_clause} WHERE symbol = ?"
            
            params = list(data.values())
            params.append(symbol)
            
            self.cursor.execute(query, params)
            self.conn.commit()
            
            self.logger.info(f"Updated stock data for symbol: {symbol}")
            return True
            
        except sqlite3.Error as e:
            self.logger.error(f"Error updating stock data: {e}")
            self.conn.rollback()
            return False
    
    def delete_stock_data(self, symbol: str) -> bool:
        """
        Delete stock data from the database.
        
        Args:
            symbol: Stock symbol to delete
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.cursor.execute("DELETE FROM stock_data WHERE symbol = ?", (symbol,))
            self.conn.commit()
            
            self.logger.info(f"Deleted stock data for symbol: {symbol}")
            return True
            
        except sqlite3.Error as e:
            self.logger.error(f"Error deleting stock data: {e}")
            self.conn.rollback()
            return False
    
    def _initialize_database(self):
        """Initialize the database with required tables."""
        try:
            # Create stock_data table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    name TEXT,
                    market_type TEXT,
                    last_price REAL,
                    volume INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create predictions table
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    predicted_price REAL NOT NULL,
                    model_name TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON stock_data(symbol)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_type ON stock_data(market_type)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_symbol ON predictions(symbol)")
            self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_pred_model ON predictions(model_name)")
            
            self.conn.commit()
            self.logger.info("Database initialized successfully")
            
        except sqlite3.Error as e:
            self.logger.error(f"Error initializing database: {e}")
            self.conn.rollback()
            raise
    
    def save_prediction(self, symbol: str, predicted_price: float, model_name: str) -> bool:
        """
        Save a prediction to the database.
        
        Args:
            symbol: Stock symbol
            predicted_price: Predicted price value
            model_name: Name of the model used for prediction
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.cursor.execute("""
                INSERT INTO predictions (symbol, predicted_price, model_name)
                VALUES (?, ?, ?)
            """, (symbol, predicted_price, model_name))
            
            self.conn.commit()
            self.logger.info(f"Saved prediction for {symbol} using {model_name}")
            return True
            
        except sqlite3.Error as e:
            self.logger.error(f"Error saving prediction: {e}")
            self.conn.rollback()
            return False
    
    def get_predictions(self, symbol: Optional[str] = None, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get predictions from the database.
        
        Args:
            symbol: Optional filter by stock symbol
            model_name: Optional filter by model name
            
        Returns:
            List of dictionaries containing predictions
        """
        try:
            conditions = []
            params = []
            
            if symbol:
                conditions.append("symbol = ?")
                params.append(symbol)
            
            if model_name:
                conditions.append("model_name = ?")
                params.append(model_name)
            
            query = "SELECT * FROM predictions"
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY timestamp DESC"
            
            self.cursor.execute(query, params)
            columns = [description[0] for description in self.cursor.description]
            results = []
            
            for row in self.cursor.fetchall():
                results.append(dict(zip(columns, row)))
            
            return results
            
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving predictions: {e}")
            return []
    
    def close(self):
        """Close the database connection."""
        self.disconnect()
        
    def __del__(self):
        """Ensure the database connection is closed when the object is destroyed."""
        self.disconnect() 