import duckdb
import glob
import os
import pandas as pd

def print_database_info(db_path):
    print(f"\n{'='*50}")
    print(f"Database: {db_path}")
    print(f"{'='*50}")
    
    try:
        con = duckdb.connect(db_path)
        
        # Get list of tables in the database
        tables = con.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'main'
        """).df()
        
        if len(tables) == 0:
            print("\nNo tables found in database.")
            return
            
        print("\nTables and Records:")
        for table in tables['table_name']:
            print(f"\n{'-'*30}")
            print(f"TABLE: {table}")
            print(f"{'-'*30}")
            
            try:
                # Check if table is empty
                row_count = con.execute(f"""
                    SELECT COUNT(*) as count
                    FROM {table}
                """).fetchone()[0]
                
                if row_count == 0:
                    # Drop empty table
                    con.execute(f"DROP TABLE {table}")
                    print(f"Deleted empty table: {table}")
                    continue
                
                # Show all records for non-empty tables
                df = con.execute(f"""
                    SELECT *
                    FROM {table}
                """).df()
                
                # Set display options for better visibility
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                pd.set_option('display.max_rows', None)
                print("\nAll Records:")
                print(df)
                print(f"\nTotal rows: {len(df)}")
                    
            except Exception as e:
                print(f"Error reading table {table}: {str(e)}")
                
        con.close()
        
    except Exception as e:
        print(f"Error connecting to database: {str(e)}")

# Specific databases to look for
target_dbs = ['stock.db', 's.db', 'stocks.db', 'forex-duckdb.db', 'stock-duckdb.db', 'nasdaq-stocks.db']

# Find all .db files recursively in current directory and subdirectories
db_files = []
for root, dirs, files in os.walk('.'):
    for file in files:
        if file in target_dbs:
            db_files.append(os.path.join(root, file))

if not db_files:
    print("None of the specified database files were found.")
    print("Looked for:", ", ".join(target_dbs))
else:
    print(f"Found {len(db_files)} of the specified database(s):")
    for db_file in db_files:
        print_database_info(db_file)
