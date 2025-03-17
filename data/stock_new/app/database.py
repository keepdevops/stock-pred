import sqlite3
from pathlib import Path

class DatabaseConnector:
    def __init__(self):
        self.connection = None
    
    def connect(self, db_path: Path):
        """Connect to a database"""
        try:
            self.connection = sqlite3.connect(db_path)
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def get_tables(self):
        """Get list of tables in database"""
        if not self.connection:
            return []
        
        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        return [row[0] for row in cursor.fetchall()] 