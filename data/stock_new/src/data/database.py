import duckdb
import logging
import os
import signal
import psutil
from pathlib import Path

class DatabaseConnector:
    def __init__(self, db_path='data/market_data.duckdb'):
        self.logger = logging.getLogger(__name__)
        self.db_path = db_path if isinstance(db_path, str) else db_path.get('path', 'data/market_data.duckdb')
        self.connection = None
        self.initialize_database()

    def release_locks(self):
        """Release any existing database locks."""
        try:
            db_path = Path(self.db_path)
            if db_path.exists():
                lock_file = Path(str(db_path) + '.lock')
                if lock_file.exists():
                    self.logger.info("Found lock file, attempting to release...")
                    try:
                        lock_file.unlink()
                        self.logger.info("Lock file removed successfully")
                    except Exception as e:
                        self.logger.error(f"Error removing lock file: {e}")
        except Exception as e:
            self.logger.error(f"Error in release_locks: {e}")

    def initialize_database(self):
        """Initialize the database connection and tables."""
        try:
            # Release any existing locks first
            self.release_locks()

            # Create data directory if it doesn't exist
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

            # Connect to database
            self.connection = duckdb.connect(self.db_path)
            self.logger.info(f"Connected to database: {self.db_path}")

            # Initialize tables
            self.connection.execute("""
                CREATE TABLE IF NOT EXISTS stock_data (
                    date DATE,
                    ticker VARCHAR,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    adj_close DOUBLE,
                    volume BIGINT,
                    PRIMARY KEY (date, ticker)
                )
            """)

            # Log table structure
            result = self.connection.execute("DESCRIBE stock_data").fetchall()
            self.logger.info(f"Table structure: {result}")

        except Exception as e:
            self.logger.error(f"Error connecting to database: {e}")
            raise

    def __del__(self):
        """Ensure connection is closed on deletion."""
        try:
            if self.connection:
                self.connection.close()
                self.logger.info("Database connection closed")
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")

    def execute_query(self, query, params=None):
        """Execute a query with optional parameters."""
        try:
            if params:
                return self.connection.execute(query, params).fetchdf()
            return self.connection.execute(query).fetchdf()
        except Exception as e:
            self.logger.error(f"Error executing query: {e}")
            raise

    def save_data(self, df, table_name='stock_data'):
        """Save DataFrame to database."""
        try:
            # Convert DataFrame to Arrow table
            arrow_table = df.to_arrow()
            
            # Create or replace table
            self.connection.execute(f"""
                INSERT OR REPLACE INTO {table_name}
                SELECT * FROM arrow_table
            """)
            self.connection.commit()
            self.logger.info(f"Saved {len(df)} rows to {table_name}")
            
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            raise 