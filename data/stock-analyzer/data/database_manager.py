"""
Database manager for handling database operations
"""
import os
import sqlite3
import duckdb
from sqlalchemy import create_engine, inspect
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd

class DatabaseManager:
    def __init__(self, data_dir=None):
        """Initialize database manager"""
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = data_dir if data_dir else self.base_dir
        print(f"Database directory set to: {self.data_dir}")
        
    def get_available_databases(self):
        """Get all available databases in the data directory"""
        databases = []
        # Check base directory for database files
        for file in os.listdir(self.data_dir):
            if file.endswith(('.db', '.sqlite', '.sqlite3', '.duckdb')):
                databases.append(file)
        print(f"Found databases in {self.data_dir}: {databases}")
        return databases
    
    def detect_db_type(self, db_path):
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
        
    def get_tables(self, db_name):
        """Get all tables in a database"""
        db_path = os.path.join(self.data_dir, db_name)
        
        # Detect database type
        db_type = self.detect_db_type(db_path)
        
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
                # Use SQLAlchemy for SQLite
                engine = create_engine(f'sqlite:///{db_path}')
                inspector = inspect(engine)
                return inspector.get_table_names()
            except SQLAlchemyError as e:
                print(f"Error getting tables: {str(e)}")
                return []
        else:
            print(f"Unknown database type for {db_name}")
            return []
    
    def get_tickers(self, db_name, table_name):
        """Get all tickers in a table"""
        db_path = os.path.join(self.data_dir, db_name)
        
        # Detect database type
        db_type = self.detect_db_type(db_path)
        
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
                # Use SQLite for sqlite databases
                engine = create_engine(f'sqlite:///{db_path}')
                # Check if table has ticker column
                inspector = inspect(engine)
                columns = [c['name'] for c in inspector.get_columns(table_name)]
                
                if 'ticker' in columns:
                    query = f"SELECT DISTINCT ticker FROM {table_name} ORDER BY ticker"
                    df = pd.read_sql(query, engine)
                    return df['ticker'].tolist()
                else:
                    print(f"No ticker column found in {table_name}")
                    return []
            except SQLAlchemyError as e:
                print(f"Error getting tickers: {str(e)}")
                return []
        else:
            print(f"Unknown database type for {db_name}")
            return []
    
    def get_data(self, db_name, table_name, ticker=None, start_date=None, end_date=None):
        """Get data from a table, optionally filtered by ticker and date range"""
        db_path = os.path.join(self.data_dir, db_name)
        
        # Detect database type
        db_type = self.detect_db_type(db_path)
        
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
                engine = create_engine(f'sqlite:///{db_path}')
                df = pd.read_sql_query(query, engine, params=params)
                return df
            else:
                print(f"Unknown database type for {db_name}")
                return pd.DataFrame()
        except Exception as e:
            print(f"Error getting data: {str(e)}")
            return pd.DataFrame() 