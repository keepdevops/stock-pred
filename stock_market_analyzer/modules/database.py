import duckdb
import pandas as pd
import logging
from datetime import datetime, timedelta

class Database:
    def __init__(self, db_path="stock_data.duckdb"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.setup_database()
        
    def setup_database(self):
        """Set up the database schema."""
        try:
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
            
        except Exception as e:
            self.logger.error(f"Error setting up database: {e}")
            raise
            
    def save_stock_data(self, data: pd.DataFrame, symbol: str):
        """Save stock data to the database."""
        try:
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