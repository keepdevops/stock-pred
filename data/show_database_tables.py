import duckdb
from datetime import datetime
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

class DatabaseInspector:
    def __init__(self, db_name='historical_market_data.db'):
        self.db_name = db_name
        try:
            self.conn = duckdb.connect(db_name, read_only=True)
            print(f"Successfully connected to {db_name}")
        except Exception as e:
            print(f"Error connecting to database: {e}")
            raise

    def show_tables(self):
        """Display all tables in the database"""
        try:
            print("\n=== Database Tables ===")
            tables = self.conn.execute("SHOW TABLES").fetchall()
            if not tables:
                print("No tables found in database")
                return
            
            print(f"Found {len(tables)} tables:")
            for table in tables:
                print(f"\n- {table[0]}")
                # Show table schema
                schema = self.conn.execute(f"DESCRIBE {table[0]}").fetchall()
                print("  Columns:")
                for col in schema:
                    print(f"    {col[0]:<20} {col[1]}")
        except Exception as e:
            print(f"Error showing tables: {e}")

    def show_table_statistics(self):
        """Display statistics for each table"""
        try:
            print("\n=== Table Statistics ===")
            tables = self.conn.execute("SHOW TABLES").fetchall()
            
            for table in tables:
                table_name = table[0]
                print(f"\nTable: {table_name}")
                print("=" * 50)
                
                # Get row count
                count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                print(f"Total Records: {count:,}")
                
                # Get date range if table has date column
                try:
                    date_range = self.conn.execute(f"""
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
                    if 'symbol' in [col[0] for col in self.conn.execute(f"DESCRIBE {table_name}").fetchall()]:
                        symbols = self.conn.execute(f"SELECT COUNT(DISTINCT symbol) FROM {table_name}").fetchone()[0]
                        print(f"Unique Symbols: {symbols:,}")
                    
                    if 'sector' in [col[0] for col in self.conn.execute(f"DESCRIBE {table_name}").fetchall()]:
                        sectors = self.conn.execute(f"SELECT COUNT(DISTINCT sector) FROM {table_name}").fetchone()[0]
                        print(f"Number of Sectors: {sectors}")
                except Exception:
                    pass
        except Exception as e:
            print(f"Error showing statistics: {e}")

    def show_sample_data(self):
        """Display sample data from each table"""
        try:
            print("\n=== Sample Data ===")
            tables = self.conn.execute("SHOW TABLES").fetchall()
            
            for table in tables:
                table_name = table[0]
                print(f"\nTable: {table_name}")
                print("=" * 50)
                
                # Get 5 sample rows
                sample = self.conn.execute(f"""
                    SELECT *
                    FROM {table_name}
                    LIMIT 5
                """).fetchdf()
                
                print(sample)
                print("\n")
        except Exception as e:
            print(f"Error showing sample data: {e}")

    def cleanup(self):
        """Close database connection"""
        try:
            self.conn.close()
            print("\nDatabase connection closed successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")

def main():
    inspector = None
    try:
        inspector = DatabaseInspector()
        
        # Show basic table information
        inspector.show_tables()
        
        # Show statistics for each table
        inspector.show_table_statistics()
        
        # Show sample data
        inspector.show_sample_data()
        
    except Exception as e:
        print(f"Fatal error: {e}")
    finally:
        if inspector:
            inspector.cleanup()

if __name__ == "__main__":
    main() 