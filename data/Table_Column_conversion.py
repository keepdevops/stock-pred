import duckdb

def rename_columns_to_ticker(database_path):
    # Connect to the DuckDB database
    conn = duckdb.connect(database_path)
    
    try:
        # Get the list of tables in the database
        tables = conn.execute("SHOW TABLES").fetchall()
        
        for table in tables:
            table_name = table[0]
            print(f"Checking table: {table_name}")
            
            # Get the columns of the current table
            columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
            column_names = [col[1] for col in columns]
            
            # Check and rename 'pair' to 'ticker'
            if 'pair' in column_names:
                print(f"Renaming 'pair' to 'ticker' in table {table_name}...")
                conn.execute(f"ALTER TABLE {table_name} RENAME COLUMN pair TO ticker;")
            
            # Check and rename 'symbol' to 'ticker'
            if 'symbol' in column_names:
                print(f"Renaming 'symbol' to 'ticker' in table {table_name}...")
                conn.execute(f"ALTER TABLE {table_name} RENAME COLUMN symbol TO ticker;")
        
        print("Column renaming completed.")
    
    except Exception as e:
        print(f"An error occurred: {e}")
    
    finally:
        # Close the connection
        conn.close()

# Specify the path to your DuckDB database
database_path = 'historical_market_data.db'

# Run the renaming function
rename_columns_to_ticker(database_path)