#!/usr/bin/env python3
print("Starting test...")

import pandas as pd
import json
import duckdb
import os
import polars as pl
import sqlite3

print("Imports successful")

def test_csv_loading():
    print("\nTesting CSV loading...")
    try:
        # Read CSV directly with pandas
        df = pd.read_csv("test_data.csv")
        print("CSV data loaded into DataFrame:")
        print(df.head())
        
        # Create DuckDB connection and table
        con = duckdb.connect("test_market_data.duckdb")
        con.execute("""
            CREATE TABLE IF NOT EXISTS technology_stocks (
                date DATE,
                ticker VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT
            )
        """)
        
        # Insert data
        con.execute("INSERT INTO technology_stocks SELECT * FROM df")
        print("CSV data loaded into DuckDB successfully")
        
        # Verify data
        result = con.execute("SELECT * FROM technology_stocks").fetchdf()
        print("\nVerifying data in DuckDB:")
        print(result.head())
        return True
        
    except Exception as e:
        print(f"Error in CSV loading: {e}")
        return False
    finally:
        if 'con' in locals():
            con.close()

def test_json_loading():
    print("\nTesting JSON loading...")
    try:
        # Read JSON file
        with open("test_data.json", "r") as f:
            json_data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(json_data)
        print("JSON data loaded into DataFrame:")
        print(df.head())
        
        # Create DuckDB connection and table
        con = duckdb.connect("test_market_data.duckdb")
        con.execute("""
            CREATE TABLE IF NOT EXISTS json_stocks (
                date DATE,
                ticker VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT
            )
        """)
        
        # Insert data
        con.execute("INSERT INTO json_stocks SELECT * FROM df")
        print("JSON data loaded into DuckDB successfully")
        
        # Verify data
        result = con.execute("SELECT * FROM json_stocks").fetchdf()
        print("\nVerifying data in DuckDB:")
        print(result.head())
        return True
        
    except Exception as e:
        print(f"Error in JSON loading: {e}")
        return False
    finally:
        if 'con' in locals():
            con.close()

def cleanup():
    print("\nCleaning up test database...")
    try:
        if os.path.exists("test_market_data.duckdb"):
            os.remove("test_market_data.duckdb")
            print("Test database removed successfully")
    except Exception as e:
        print(f"Error during cleanup: {e}")

def main():
    try:
        # Run tests
        csv_success = test_csv_loading()
        json_success = test_json_loading()
        
        # Print summary
        print("\nTest Summary:")
        print(f"CSV Loading: {'✓ Passed' if csv_success else '✗ Failed'}")
        print(f"JSON Loading: {'✓ Passed' if json_success else '✗ Failed'}")
        
    finally:
        cleanup()

if __name__ == "__main__":
    main()

loader = DataLoader({'source': 'sqlite', 'db_path': 'path/to/database.db'})
df, error = loader.fetch_stock_data('AAPL')

# CSV
df, error = loader.load_from_file('data.csv', 'csv')

# JSON
df, error = loader.load_from_file('data.json', 'json')

# Parquet
df, error = loader.load_from_file('data.parquet', 'parquet')

is_valid, error = loader.validate_symbol('AAPL')

info, error = loader.get_company_info('AAPL')

data_loader = DataLoader(config)
data_dict = data_loader.load_from_directory("/path/to/stock/data")

query = "SELECT * FROM stock_data WHERE symbol = 'AAPL'"
df = data_loader.load_from_sqlite("stock_data.db", query)

symbols = data_loader.list_available_symbols("/path/to/stock/data")

info = data_loader.get_file_info("/path/to/stock/data/AAPL_2024.csv")

polars_df = pl.read_csv("stock_data.csv")
pandas_df = data_loader.load_from_polars(polars_df) 