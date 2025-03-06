"""
Database connection and query functionality
"""
import glob
import os
from sqlalchemy import create_engine, text
import pandas as pd

def find_databases(data_dir=None):
    """Find all database files in the specified directory"""
    if data_dir is None:
        # Default to current directory
        data_dir = os.getcwd()
    
    # If we're in the stock-analyzer directory, look in the parent's data directory
    if os.path.basename(data_dir) == 'stock-analyzer':
        data_dir = os.path.join(os.path.dirname(data_dir), 'data')
    
    # Look for database files in the specified directory
    db_files = glob.glob(os.path.join(data_dir, '*.db')) + glob.glob(os.path.join(data_dir, '*.duckdb'))
    
    # Return just the filenames, not the full paths
    return [os.path.basename(db) for db in db_files]

def create_connection(db_name, data_dir=None):
    """Create a database connection using SQLAlchemy"""
    try:
        if data_dir is None:
            # Default to current directory
            data_dir = os.getcwd()
        
        # If we're in the stock-analyzer directory, look in the parent's data directory
        if os.path.basename(data_dir) == 'stock-analyzer':
            data_dir = os.path.join(os.path.dirname(data_dir), 'data')
        
        # Full path to the database file
        db_path = os.path.join(data_dir, db_name)
        
        if db_name.endswith('.duckdb'):
            engine = create_engine(f"duckdb:///{db_path}")
        else:
            engine = create_engine(f"sqlite:///{db_path}")
        return engine
    except Exception as e:
        print(f"Error connecting to database {db_name}: {e}")
        return None

def get_tables(db_name, data_dir=None):
    """Get list of tables from the specified database"""
    engine = create_connection(db_name, data_dir)
    if not engine:
        return []
    
    try:
        with engine.connect() as conn:
            # For SQLite
            if db_name.endswith('.db'):
                query = text("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in conn.execute(query)]
            # For DuckDB
            elif db_name.endswith('.duckdb'):
                query = text("SELECT table_name FROM information_schema.tables WHERE table_schema='main'")
                tables = [row[0] for row in conn.execute(query)]
            else:
                tables = []
        return tables
    except Exception as e:
        print(f"Error getting tables: {e}")
        return []
    finally:
        engine.dispose()

def get_tickers(db_name, table_name, data_dir=None):
    """Get list of tickers from the specified table"""
    engine = create_connection(db_name, data_dir)
    if not engine:
        return []
    
    try:
        query = text(f"SELECT DISTINCT ticker FROM {table_name} ORDER BY ticker")
        with engine.connect() as conn:
            result = conn.execute(query)
            tickers = [row[0] for row in result]
        return tickers
    except Exception as e:
        print(f"Error getting tickers: {e}")
        return []
    finally:
        engine.dispose()

def fetch_ticker_data(db_name, table_name, tickers, data_dir=None):
    """Fetch data for the specified tickers"""
    engine = create_connection(db_name, data_dir)
    if not engine:
        return None
    
    try:
        placeholders = ', '.join([':ticker' + str(i) for i in range(len(tickers))])
        query = text(f"SELECT * FROM {table_name} WHERE ticker IN ({placeholders})")
        params = {f'ticker{i}': ticker for i, ticker in enumerate(tickers)}
        
        with engine.connect() as conn:
            result = conn.execute(query, params)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
        return df
    except Exception as e:
        print(f"Error fetching ticker data: {e}")
        return None
    finally:
        engine.dispose()
