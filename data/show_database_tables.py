import duckdb
from datetime import datetime
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

class DatabaseInspector:
    def __init__(self, db_names=['historical_market_data.db', 'other_db.db']):
        self.db_names = db_names if isinstance(db_names, list) else [db_names]
        self.connections = {}
        
        for db_name in self.db_names:
            try:
                conn = duckdb.connect(db_name, read_only=True)
                self.connections[db_name] = conn
                print(f"Successfully connected to {db_name}")
            except Exception as e:
                print(f"Error connecting to database {db_name}: {e}")

    def show_tables(self):
        """Display all tables in all databases"""
        try:
            for db_name, conn in self.connections.items():
                print(f"\n=== Database: {db_name} ===")
                tables = conn.execute("SHOW TABLES").fetchall()
                if not tables:
                    print("No tables found in database")
                    continue
                
                print(f"Found {len(tables)} tables:")
                for table in tables:
                    print(f"\n- {table[0]}")
                    # Show table schema
                    schema = conn.execute(f"DESCRIBE {table[0]}").fetchall()
                    print("  Columns:")
                    for col in schema:
                        print(f"    {col[0]:<20} {col[1]}")
        except Exception as e:
            print(f"Error showing tables: {e}")

    def show_table_statistics(self):
        """Display statistics for each table"""
        try:
            print("\n=== Table Statistics ===")
            for db_name, conn in self.connections.items():
                print(f"\n=== Database: {db_name} ===")
                tables = conn.execute("SHOW TABLES").fetchall()
                
                for table in tables:
                    table_name = table[0]
                    print(f"\nTable: {table_name}")
                    print("=" * 50)
                    
                    # Get row count
                    count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                    print(f"Total Records: {count:,}")
                    
                    # Get date range if table has date column
                    try:
                        date_range = conn.execute(f"""
                            SELECT 
                                MIN(date) as earliest,
                                MAX(date) as latest
                            FROM {table_name}
                        """).fetchone()
                        print(f"Date Range: {date_range[0]} to {date_range[1]}")
                    except Exception:
                        pass  # Table might not have date column
                    
                    # Get unique values for key columns
                    try:
                        if 'symbol' in [col[0] for col in conn.execute(f"DESCRIBE {table_name}").fetchall()]:
                            symbols = conn.execute(f"SELECT COUNT(DISTINCT symbol) FROM {table_name}").fetchone()[0]
                            print(f"Unique Symbols: {symbols:,}")
                        
                        if 'sector' in [col[0] for col in conn.execute(f"DESCRIBE {table_name}").fetchall()]:
                            sectors = conn.execute(f"SELECT COUNT(DISTINCT sector) FROM {table_name}").fetchone()[0]
                            print(f"Number of Sectors: {sectors}")
                    except Exception:
                        pass
        except Exception as e:
            print(f"Error showing statistics: {e}")

    def show_sample_data(self):
        """Display sample data from each table"""
        try:
            print("\n=== Sample Data ===")
            for db_name, conn in self.connections.items():
                print(f"\n=== Database: {db_name} ===")
                tables = conn.execute("SHOW TABLES").fetchall()
                
                for table in tables:
                    table_name = table[0]
                    print(f"\nTable: {table_name}")
                    print("=" * 50)
                    
                    # Get 5 sample rows
                    sample = conn.execute(f"""
                        SELECT *
                        FROM {table_name}
                        LIMIT 5
                    """).fetchdf()
                    
                    print(sample)
                    print("\n")
        except Exception as e:
            print(f"Error showing sample data: {e}")

    def cleanup(self):
        """Close all database connections"""
        try:
            for db_name, conn in self.connections.items():
                conn.close()
                print(f"Database connection to {db_name} closed successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    # List all your database files here
    db_files = [
        'historical_market_data.db',
        # Add other database files as needed
    ]
    
    inspector = None
    try:
        inspector = DatabaseInspector(db_files)
        inspector.show_tables()
        inspector.show_sample_data()
        
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        if inspector:
            inspector.cleanup()

if __name__ == "__main__":
    main() 