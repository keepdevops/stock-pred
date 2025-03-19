import logging
from pathlib import Path
from typing import Optional, Dict, List
import duckdb
import pandas as pd

class DatabaseConnector:
    """Handles database connections and operations."""
    
    def __init__(
        self,
        db_path: str,
        logger: Optional[logging.Logger] = None
    ):
        self.db_path = db_path
        self.logger = logger or logging.getLogger(__name__)
        self.conn = None
        self.connect()
    
    def connect(self) -> None:
        """Establish database connection."""
        try:
            # Create directory if it doesn't exist
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
            
            # Connect to database
            self.conn = duckdb.connect(database=self.db_path)
            self.logger.info(f"Connected to database: {self.db_path}")
            
            # Initialize tables if they don't exist
            self._initialize_tables()
            
        except Exception as e:
            self.logger.error(f"Error connecting to database: {str(e)}")
            raise
    
    def _initialize_tables(self) -> None:
        """Initialize database tables."""
        try:
            # Create stock data table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    date TIMESTAMP,
                    ticker VARCHAR,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    PRIMARY KEY (date, ticker)
                )
            """)
            
            # Create predictions table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    date TIMESTAMP,
                    ticker VARCHAR,
                    predicted_price DOUBLE,
                    confidence DOUBLE,
                    model_type VARCHAR,
                    PRIMARY KEY (date, ticker, model_type)
                )
            """)
            
            # Create technical indicators table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    date TIMESTAMP,
                    ticker VARCHAR,
                    indicator_name VARCHAR,
                    value DOUBLE,
                    PRIMARY KEY (date, ticker, indicator_name)
                )
            """)
            
            # Create trading history table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_history (
                    trade_id BIGINT PRIMARY KEY,
                    date TIMESTAMP,
                    ticker VARCHAR,
                    action VARCHAR,
                    quantity DOUBLE,
                    price DOUBLE,
                    total_value DOUBLE
                )
            """)
            
            # Create sequence for trade_id
            self.conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS trading_history_trade_id_seq
                START 1 INCREMENT 1
            """)
            
            self.logger.info("Database tables initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing tables: {str(e)}")
            raise
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")
    
    def save_ticker_data(
        self,
        ticker: str,
        data: pd.DataFrame,
        realtime: bool = False
    ) -> None:
        """Save ticker data to database."""
        try:
            # Prepare data
            df = data.copy()
            df['ticker'] = ticker
            
            # Insert data
            if realtime:
                # For realtime data, we might want to update existing records
                self.conn.execute("""
                    INSERT OR REPLACE INTO stock_data
                    SELECT * FROM df
                """)
            else:
                # For historical data, ignore duplicates
                self.conn.execute("""
                    INSERT OR IGNORE INTO stock_data
                    SELECT * FROM df
                """)
            
            self.logger.info(f"Saved data for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error saving data for {ticker}: {str(e)}")
            raise
    
    def save_trade(
        self,
        ticker: str,
        action: str,
        quantity: float,
        price: float
    ) -> None:
        """Save trade to history."""
        try:
            total_value = quantity * price
            
            self.conn.execute("""
                INSERT INTO trading_history (
                    trade_id, date, ticker, action, quantity, price, total_value
                )
                SELECT 
                    nextval('trading_history_trade_id_seq'),
                    CURRENT_TIMESTAMP,
                    ?, ?, ?, ?, ?
            """, [ticker, action, quantity, price, total_value])
            
            self.logger.info(f"Saved trade for {ticker}")
            
        except Exception as e:
            self.logger.error(f"Error saving trade: {str(e)}")
            raise
    
    def get_ticker_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Retrieve ticker data from database."""
        try:
            query = "SELECT * FROM stock_data WHERE ticker = ?"
            params = [ticker]
            
            if start_date:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date"
            
            result = self.conn.execute(query, params).fetchdf()
            return result if not result.empty else None
            
        except Exception as e:
            self.logger.error(f"Error retrieving data for {ticker}: {str(e)}")
            return None
    
    def get_trading_history(
        self,
        ticker: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Retrieve trading history."""
        try:
            query = "SELECT * FROM trading_history"
            params = []
            
            if ticker:
                query += " WHERE ticker = ?"
                params.append(ticker)
            
            if start_date:
                query += " AND date >= ?" if ticker else " WHERE date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND date <= ?"
                params.append(end_date)
            
            query += " ORDER BY date DESC"
            
            return self.conn.execute(query, params).fetchdf()
            
        except Exception as e:
            self.logger.error(f"Error retrieving trading history: {str(e)}")
            return pd.DataFrame()
    
    def get_tables(self) -> List[str]:
        """Get list of tables in the database."""
        try:
            tables = self.conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'main'
            """).fetchdf()
            return tables['table_name'].tolist()
        except Exception:
            # Try SQLite syntax if DuckDB fails
            try:
                tables = self.conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table'
                """).fetchdf()
                return tables['name'].tolist()
            except Exception as e:
                self.logger.error(f"Error getting tables: {str(e)}")
                return []
    
    def initialize_tables(self) -> None:
        """Initialize database tables."""
        try:
            # Create stock data table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    date TIMESTAMP,
                    ticker VARCHAR,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    PRIMARY KEY (date, ticker)
                )
            """)
            
            # Create predictions table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    date TIMESTAMP,
                    ticker VARCHAR,
                    predicted_price DOUBLE,
                    confidence DOUBLE,
                    model_type VARCHAR,
                    PRIMARY KEY (date, ticker, model_type)
                )
            """)
            
            # Create technical indicators table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS technical_indicators (
                    date TIMESTAMP,
                    ticker VARCHAR,
                    indicator_name VARCHAR,
                    value DOUBLE,
                    PRIMARY KEY (date, ticker, indicator_name)
                )
            """)
            
            # Create trading history table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_history (
                    trade_id BIGINT PRIMARY KEY,
                    date TIMESTAMP,
                    ticker VARCHAR,
                    action VARCHAR,
                    quantity DOUBLE,
                    price DOUBLE,
                    total_value DOUBLE
                )
            """)
            
            # Create sequence for trade_id
            self.conn.execute("""
                CREATE SEQUENCE IF NOT EXISTS trading_history_trade_id_seq
                START 1 INCREMENT 1
            """)
            
            self.logger.info("Database tables initialized")
            
        except Exception as e:
            self.logger.error(f"Error initializing tables: {str(e)}")
            raise 