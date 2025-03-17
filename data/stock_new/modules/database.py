from sqlalchemy import create_engine, text
import pandas as pd
from pathlib import Path

class DatabaseConnector:
    def __init__(self):
        self.engine = None
        self.current_db = None
    
    def create_connection(self, db_path: Path) -> bool:
        """Create SQLAlchemy engine for DuckDB connection."""
        try:
            self.engine = create_engine(f"duckdb:///{db_path}")
            self.current_db = db_path
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def get_tables(self) -> list[str]:
        """Get all tables in current database."""
        if not self.engine:
            return []
        
        with self.engine.connect() as conn:
            result = conn.execute(text("SELECT table_name FROM information_schema.tables"))
            return [row[0] for row in result]
    
    def load_tickers(self, table: str) -> pd.DataFrame:
        """Load ticker data from specified table."""
        if not self.engine:
            return pd.DataFrame()
        
        query = f"SELECT * FROM {table} ORDER BY date"
        return pd.read_sql(query, self.engine) 