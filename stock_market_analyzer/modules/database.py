import sqlite3
import logging
from pathlib import Path
import pandas as pd

class DatabaseConnector:
    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.db_path = Path(config.get('path', 'stocks.db'))
        self.db_type = config.get('type', 'sqlite')
        
        self.setup_database()
        
    def setup_database(self):
        """Set up the database and create necessary tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create stocks table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS stocks (
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
                
                # Create trading history table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        type TEXT,
                        size REAL,
                        price REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                self.logger.info("Database setup completed successfully")
                
        except Exception as e:
            self.logger.error(f"Database setup failed: {e}")
            raise
            
    def insert_stock_data(self, symbol: str, data: pd.DataFrame):
        """Insert stock data into the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Convert DataFrame to SQL-friendly format
                data['symbol'] = symbol
                data.to_sql('stocks', conn, if_exists='append', index=False)
                
            self.logger.info(f"Successfully inserted data for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error inserting stock data: {e}")
            raise
            
    def get_stock_data(self, symbol: str, start_date: str = None, end_date: str = None):
        """Retrieve stock data from the database."""
        try:
            query = "SELECT * FROM stocks WHERE symbol = ?"
            params = [symbol]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
                
            query += " ORDER BY date"
            
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(query, conn, params=params)
                
        except Exception as e:
            self.logger.error(f"Error retrieving stock data: {e}")
            raise
            
    def record_trade(self, symbol: str, trade_type: str, size: float, price: float):
        """Record a trade in the trading history."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO trading_history (symbol, type, size, price)
                    VALUES (?, ?, ?, ?)
                """, (symbol, trade_type, size, price))
                conn.commit()
                
            self.logger.info(f"Trade recorded: {trade_type} {size} {symbol} @ {price}")
            
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
            raise
            
    def get_trading_history(self, symbol: str = None):
        """Retrieve trading history from the database."""
        try:
            query = "SELECT * FROM trading_history"
            params = []
            
            if symbol:
                query += " WHERE symbol = ?"
                params.append(symbol)
                
            query += " ORDER BY timestamp DESC"
            
            with sqlite3.connect(self.db_path) as conn:
                return pd.read_sql_query(query, conn, params=params)
                
        except Exception as e:
            self.logger.error(f"Error retrieving trading history: {e}")
            raise
            
    def close(self):
        """Close any open database connections."""
        # SQLite connections are automatically closed after each transaction
        pass 