import sqlite3
import os
from pathlib import Path

def init_database():
    """Initialize the SQLite database with the correct schema."""
    # Get the project root directory
    project_root = Path(__file__).parent.parent.parent
    
    # Define the database path
    db_path = project_root / "data" / "stock_data.db"
    
    # Create the data directory if it doesn't exist
    os.makedirs(project_root / "data", exist_ok=True)
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Create the stock_data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                name TEXT,
                market_type TEXT,
                last_price REAL,
                volume INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON stock_data(symbol)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_market_type ON stock_data(market_type)")
        
        # Commit the changes
        conn.commit()
        print(f"Database initialized at: {db_path}")
        
    except sqlite3.Error as e:
        print(f"Error initializing database: {e}")
        conn.rollback()
        
    finally:
        # Close the connection
        conn.close()

if __name__ == "__main__":
    init_database() 