import duckdb
import os

# Define database path - store in same directory as script
FOREX_DB = os.path.join(os.path.dirname(__file__), "forex-duckdb.db")

def get_connection(db_path):
    """Create and return a DuckDB connection"""
    return duckdb.connect(db_path)

def create_tables(conn):
    """Create required tables if they don't exist"""
    try:
        # Create market_data table for both forex and stock data
        conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                date DATE,
                ticker VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                adj_close DOUBLE,
                volume DOUBLE,
                type VARCHAR
            )
        """)
        
        # Create indices for common queries
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_date_ticker 
            ON market_data(date, ticker)
        """)
        
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_type 
            ON market_data(ticker, type)
        """)

    except Exception as e:
        print(f"Error creating tables: {str(e)}")
        raise 