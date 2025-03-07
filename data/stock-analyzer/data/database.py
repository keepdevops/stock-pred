"""
Database utility functions
"""
import os
import sqlite3
import duckdb
import pandas as pd

def find_databases(data_dir):
    """Find all database files in the data directory"""
    databases = []
    for file in os.listdir(data_dir):
        if file.endswith(('.db', '.sqlite', '.sqlite3', '.duckdb')):
            databases.append(file)
    print(f"Found databases in {data_dir}: {databases}")
    return databases

def detect_db_type(db_path):
    """Detect if a database is SQLite or DuckDB"""
    # Check for DuckDB based on file extension
    if db_path.endswith('.duckdb'):
        return 'duckdb'
    
    # Try opening with DuckDB
    try:
        conn = duckdb.connect(db_path, read_only=True)
        # Try a simple query to confirm it's a valid DuckDB database
        conn.execute("SELECT 1")
        conn.close()
        return 'duckdb'
    except:
        pass
    
    # Try opening with SQLite as a fallback
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("SELECT 1")
        conn.close()
        return 'sqlite'
    except:
        # If both fail, return unknown
        return 'unknown'

def get_tables(db_name, data_dir):
    """Get all tables in a database"""
    db_path = os.path.join(data_dir, db_name)
    
    # Detect database type
    db_type = detect_db_type(db_path)
    
    if db_type == 'duckdb':
        try:
            conn = duckdb.connect(db_path)
            # Query for listing tables in DuckDB
            tables = conn.execute("SHOW TABLES").fetchall()
            conn.close()
            # Extract table names from the result
            return [table[0] for table in tables]
        except Exception as e:
            print(f"Error getting tables from DuckDB: {str(e)}")
            return []
    elif db_type == 'sqlite':
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = cursor.fetchall()
            conn.close()
            return [table[0] for table in tables]
        except Exception as e:
            print(f"Error getting tables from SQLite: {str(e)}")
            return []
    else:
        print(f"Unknown database type for {db_name}")
        return []

def get_tickers(db_name, table_name, data_dir):
    """Get all tickers in a table"""
    db_path = os.path.join(data_dir, db_name)
    
    # Detect database type
    db_type = detect_db_type(db_path)
    
    if db_type == 'duckdb':
        try:
            conn = duckdb.connect(db_path)
            # Check if ticker column exists
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            column_names = [col[1] for col in columns]
            
            # If 'ticker' column exists, query distinct tickers
            if 'ticker' in column_names:
                result = conn.execute(f"SELECT DISTINCT ticker FROM {table_name} ORDER BY ticker").fetchall()
                conn.close()
                return [row[0] for row in result]
            else:
                print(f"No ticker column found in {table_name}")
                conn.close()
                return []
        except Exception as e:
            print(f"Error getting tickers from DuckDB: {str(e)}")
            return []
    elif db_type == 'sqlite':
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check if table has ticker column
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            if 'ticker' in column_names:
                cursor.execute(f"SELECT DISTINCT ticker FROM {table_name} ORDER BY ticker")
                tickers = cursor.fetchall()
                conn.close()
                return [ticker[0] for ticker in tickers]
            else:
                print(f"No ticker column found in {table_name}")
                conn.close()
                return []
        except Exception as e:
            print(f"Error getting tickers from SQLite: {str(e)}")
            return []
    else:
        print(f"Unknown database type for {db_name}")
        return []

def get_data(db_name, table_name, ticker=None, start_date=None, end_date=None, data_dir=None):
    """Get data from a table, optionally filtered by ticker and date range"""
    if data_dir is None:
        # Default to the parent directory if no data_dir is provided
        data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    db_path = os.path.join(data_dir, db_name)
    
    # Detect database type
    db_type = detect_db_type(db_path)
    
    try:
        query = f"SELECT * FROM {table_name}"
        params = []
        
        # Add filters
        filters = []
        if ticker:
            filters.append("ticker = ?")
            params.append(ticker)
        if start_date:
            filters.append("date >= ?")
            params.append(start_date)
        if end_date:
            filters.append("date <= ?")
            params.append(end_date)
            
        if filters:
            query += " WHERE " + " AND ".join(filters)
            
        query += " ORDER BY date"
        
        if db_type == 'duckdb':
            conn = duckdb.connect(db_path)
            # DuckDB uses a different parameter syntax
            for i, param in enumerate(params):
                query = query.replace("?", f"${i+1}", 1)
            df = conn.execute(query, params).fetchdf()
            conn.close()
            return df
        elif db_type == 'sqlite':
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df
        else:
            print(f"Unknown database type for {db_name}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error getting data: {str(e)}")
        return pd.DataFrame()
