from typing import List, Optional, Dict, Any
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path
import duckdb
import logging
from datetime import datetime

class DatabaseConnector:
    """Handles database connections and queries for stock data."""
    
    def __init__(self):
        self.engine = None
        self.current_db: Optional[Path] = None
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def create_connection(self, db_path: Path) -> bool:
        """Create SQLAlchemy engine for DuckDB connection."""
        try:
            self.engine = create_engine(f"duckdb:///{db_path}")
            self.current_db = db_path
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.logger.info(f"Successfully connected to database: {db_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Connection error: {e}")
            return False
    
    def get_tables(self) -> List[str]:
        """Get all tables in current database."""
        if not self.engine:
            self.logger.warning("No active database connection")
            return []
        
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'main'"
                ))
                return [row[0] for row in result]
        except Exception as e:
            self.logger.error(f"Error getting tables: {e}")
            return []
    
    def load_tickers(self, table: str) -> pd.DataFrame:
        """Load ticker data from specified table."""
        if not self.engine:
            self.logger.warning("No active database connection")
            return pd.DataFrame()
        
        try:
            query = f"""
            SELECT date, open, high, low, close, adj_close, volume, ticker, updated_at
            FROM {table}
            ORDER BY date, ticker
            """
            return pd.read_sql(query, self.engine)
            
        except Exception as e:
            self.logger.error(f"Error loading tickers from {table}: {e}")
            return pd.DataFrame()
    
    def get_unique_tickers(self, table: str) -> List[str]:
        """Get list of unique tickers in a table."""
        if not self.engine:
            return []
        
        try:
            query = f"SELECT DISTINCT ticker FROM {table} ORDER BY ticker"
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                return [row[0] for row in result]
        except Exception as e:
            self.logger.error(f"Error getting unique tickers: {e}")
            return []
    
    def save_training_results(
        self,
        model_name: str,
        ticker: str,
        metrics: Dict[str, float],
        predictions: pd.DataFrame
    ) -> bool:
        """Save training results to training_data.duckdb."""
        try:
            training_db = self.current_db.parent / "training_data.duckdb"
            train_engine = create_engine(f"duckdb:///{training_db}")
            
            # Save metrics
            metrics_df = pd.DataFrame([{
                'model_name': model_name,
                'ticker': ticker,
                'timestamp': datetime.now(),
                **metrics
            }])
            
            metrics_df.to_sql(
                'training_metrics',
                train_engine,
                if_exists='append',
                index=False
            )
            
            # Save predictions
            predictions['model_name'] = model_name
            predictions['ticker'] = ticker
            predictions['timestamp'] = datetime.now()
            
            predictions.to_sql(
                'predictions',
                train_engine,
                if_exists='append',
                index=False
            )
            
            self.logger.info(f"Saved training results for {ticker} using {model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving training results: {e}")
            return False
    
    def create_sector_tables(self, sector_mappings: Dict[str, List[str]]) -> bool:
        """Create tables for different market sectors."""
        if not self.engine:
            return False
        
        try:
            with self.engine.connect() as conn:
                # Create sector tables if they don't exist
                for sector, tickers in sector_mappings.items():
                    table_name = f"{sector.lower()}_stocks"
                    
                    # Create table
                    conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        date DATETIME,
                        open FLOAT,
                        high FLOAT,
                        low FLOAT,
                        close FLOAT,
                        adj_close FLOAT,
                        volume FLOAT,
                        ticker VARCHAR,
                        updated_at DATETIME
                    )
                    """))
                    
                    # Create indexes
                    conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS idx_{table_name}_date_ticker 
                    ON {table_name}(date, ticker)
                    """))
                    
                    self.logger.info(f"Created sector table: {table_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating sector tables: {e}")
            return False
    
    def update_stock_data(
        self,
        df: pd.DataFrame,
        table: str,
        if_exists: str = 'append'
    ) -> bool:
        """Update stock data in specified table."""
        if not self.engine:
            return False
        
        try:
            # Add updated_at timestamp
            df['updated_at'] = datetime.now()
            
            # Write to database
            df.to_sql(
                table,
                self.engine,
                if_exists=if_exists,
                index=False,
                method='multi',
                chunksize=1000
            )
            
            self.logger.info(f"Updated {len(df)} rows in {table}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating stock data: {e}")
            return False
    
    def get_latest_data(
        self,
        table: str,
        ticker: str,
        days: int = 30
    ) -> pd.DataFrame:
        """Get latest N days of data for a ticker."""
        if not self.engine:
            return pd.DataFrame()
        
        try:
            query = f"""
            SELECT *
            FROM {table}
            WHERE ticker = :ticker
            ORDER BY date DESC
            LIMIT :days
            """
            
            return pd.read_sql(
                query,
                self.engine,
                params={'ticker': ticker, 'days': days}
            )
            
        except Exception as e:
            self.logger.error(f"Error getting latest data: {e}")
            return pd.DataFrame()
    
    def cleanup_old_data(self, days_to_keep: int = 365) -> bool:
        """Remove data older than specified days."""
        if not self.engine:
            return False
        
        try:
            with self.engine.connect() as conn:
                # Get all tables
                tables = self.get_tables()
                
                for table in tables:
                    conn.execute(text(f"""
                    DELETE FROM {table}
                    WHERE date < DATEADD('day', -{days_to_keep}, CURRENT_DATE)
                    """))
                    
                self.logger.info(f"Cleaned up old data (keeping {days_to_keep} days)")
                return True
                
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            return False 