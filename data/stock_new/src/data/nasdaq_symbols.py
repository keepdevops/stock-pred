import pandas as pd
import duckdb
import logging
from pathlib import Path
from datetime import datetime
import requests
from typing import Optional, List, Dict, Any
import json

logger = logging.getLogger(__name__)

class NasdaqSymbolManager:
    """Manages NASDAQ symbols in a DuckDB database."""
    
    def __init__(self, db_path: str = None, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.db_path = db_path or Path('data/nasdaq_symbols.duckdb')
        self.db_path = Path(self.db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = None
        self.initialize_database()

    def initialize_database(self):
        """Initialize the database and create necessary tables."""
        try:
            self.conn = duckdb.connect(str(self.db_path))
            
            # Create symbols table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS nasdaq_symbols (
                    symbol VARCHAR PRIMARY KEY,
                    name VARCHAR,
                    market_cap DOUBLE,
                    country VARCHAR,
                    ipo_year INTEGER,
                    volume BIGINT,
                    sector VARCHAR,
                    industry VARCHAR,
                    last_updated TIMESTAMP
                )
            """)
            
            # Create symbol_history table for tracking changes
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS symbol_history (
                    symbol VARCHAR,
                    field VARCHAR,
                    old_value VARCHAR,
                    new_value VARCHAR,
                    change_date TIMESTAMP,
                    PRIMARY KEY (symbol, field, change_date)
                )
            """)
            
            self.logger.info("Database initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise

    def update_symbols(self, df: pd.DataFrame):
        """Update symbols in the database."""
        try:
            # Add last_updated timestamp
            df['last_updated'] = datetime.now()
            
            # Create temporary table
            self.conn.execute("CREATE TEMPORARY TABLE IF NOT EXISTS temp_symbols AS SELECT * FROM nasdaq_symbols WHERE 1=0")
            
            # Insert new data
            self.conn.execute("INSERT INTO temp_symbols SELECT * FROM df")
            
            # Merge into main table
            self.conn.execute("""
                INSERT OR REPLACE INTO nasdaq_symbols 
                SELECT * FROM temp_symbols
            """)
            
            # Cleanup
            self.conn.execute("DROP TABLE temp_symbols")
            
            self.logger.info(f"Updated {len(df)} symbols")
            
        except Exception as e:
            self.logger.error(f"Error updating symbols: {e}")
            raise

    def _track_symbol_changes(self, old_df: pd.DataFrame, new_df: pd.DataFrame):
        """Track changes in symbol data."""
        try:
            changes = []
            tracked_fields = ['name', 'market_cap', 'sector', 'industry', 'volume']
            
            for symbol in new_df['symbol'].unique():
                old_row = old_df[old_df['symbol'] == symbol]
                new_row = new_df[new_df['symbol'] == symbol]
                
                if old_row.empty:
                    continue
                    
                for field in tracked_fields:
                    old_value = str(old_row[field].iloc[0])
                    new_value = str(new_row[field].iloc[0])
                    
                    if old_value != new_value:
                        changes.append({
                            'symbol': symbol,
                            'field': field,
                            'old_value': old_value,
                            'new_value': new_value,
                            'change_date': datetime.now()
                        })
            
            if changes:
                changes_df = pd.DataFrame(changes)
                self.conn.execute("INSERT INTO symbol_history SELECT * FROM changes_df")
                self.logger.info(f"Recorded {len(changes)} symbol changes")
                
        except Exception as e:
            self.logger.error(f"Error tracking symbol changes: {e}")
            raise

    def get_all_symbols(self) -> pd.DataFrame:
        """Get all symbols from the database."""
        try:
            return self.conn.execute("""
                SELECT * FROM nasdaq_symbols 
                ORDER BY symbol
            """).fetchdf()
        except Exception as e:
            self.logger.error(f"Error getting symbols: {e}")
            return pd.DataFrame()

    def get_filtered_symbols(self, min_market_cap: float = None, min_volume: int = None) -> pd.DataFrame:
        """Get filtered symbols."""
        try:
            query = "SELECT * FROM nasdaq_symbols WHERE 1=1"
            params = []
            
            if min_market_cap is not None:
                query += " AND market_cap >= ?"
                params.append(min_market_cap)
            
            if min_volume is not None:
                query += " AND volume >= ?"
                params.append(min_volume)
            
            return self.conn.execute(query, params).fetchdf()
            
        except Exception as e:
            self.logger.error(f"Error getting filtered symbols: {e}")
            return pd.DataFrame()

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific symbol."""
        try:
            result = self.conn.execute("""
                SELECT * FROM nasdaq_symbols 
                WHERE symbol = ?
            """, [symbol]).fetchdf()
            
            if result.empty:
                return None
                
            return result.iloc[0].to_dict()
            
        except Exception as e:
            self.logger.error(f"Error getting symbol info: {e}")
            return None

    def get_symbol_history(self, symbol: str) -> pd.DataFrame:
        """Get change history for a symbol."""
        try:
            return self.conn.execute("""
                SELECT * FROM symbol_history 
                WHERE symbol = ? 
                ORDER BY change_date DESC
            """, [symbol]).fetchdf()
            
        except Exception as e:
            self.logger.error(f"Error getting symbol history: {e}")
            return pd.DataFrame()

    def get_sectors(self) -> List[str]:
        """Get list of unique sectors."""
        try:
            result = self.conn.execute("""
                SELECT DISTINCT sector 
                FROM nasdaq_symbols 
                WHERE sector != '' 
                ORDER BY sector
            """).fetchdf()
            return result['sector'].tolist()
            
        except Exception as e:
            self.logger.error(f"Error getting sectors: {e}")
            return []

    def get_countries(self) -> List[str]:
        """Get list of unique countries."""
        try:
            result = self.conn.execute("""
                SELECT DISTINCT country 
                FROM nasdaq_symbols 
                WHERE country != '' 
                ORDER BY country
            """).fetchdf()
            return result['country'].tolist()
            
        except Exception as e:
            self.logger.error(f"Error getting countries: {e}")
            return []

    def export_symbols(self, output_path: str):
        """Export symbols to a CSV file."""
        try:
            df = self.get_all_symbols()
            df.to_csv(output_path, index=False)
            self.logger.info(f"Exported {len(df)} symbols to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting symbols: {e}")
            raise

    def close(self):
        """Close the database connection."""
        try:
            if self.conn:
                self.conn.close()
                self.conn = None
                self.logger.info("Database connection closed")
                
        except Exception as e:
            self.logger.error(f"Error closing database: {e}")

def load_nasdaq_symbols() -> List[str]:
    """Load NASDAQ symbols from the most recent screener file."""
    try:
        data_dir = Path('data')
        screener_files = list(data_dir.glob('nasdaq_screener_*.csv'))
        
        if not screener_files:
            logger.warning("No NASDAQ screener files found")
            return []
            
        # Get most recent file
        latest_file = max(screener_files, key=lambda x: x.stat().st_mtime)
        
        # Read CSV
        df = pd.read_csv(latest_file)
        
        # Extract and clean symbols
        if 'Symbol' in df.columns:
            symbols = df['Symbol'].astype(str).str.strip().unique().tolist()
            logger.info(f"Loaded {len(symbols)} NASDAQ symbols")
            return symbols
        else:
            logger.error("Symbol column not found in screener file")
            return []
            
    except Exception as e:
        logger.error(f"Error loading NASDAQ symbols: {e}")
        return []

# Initialize the symbols list
NASDAQ_SYMBOLS = []  # Initialize empty first
try:
    NASDAQ_SYMBOLS = load_nasdaq_symbols()
except Exception as e:
    logger.error(f"Error initializing NASDAQ symbols: {e}")

def get_nasdaq_symbols() -> List[str]:
    """Get the loaded NASDAQ symbols."""
    return NASDAQ_SYMBOLS

def refresh_nasdaq_symbols() -> List[str]:
    """Refresh the NASDAQ symbols list."""
    global NASDAQ_SYMBOLS
    NASDAQ_SYMBOLS = load_nasdaq_symbols()
    return NASDAQ_SYMBOLS

def filter_symbols(min_market_cap: Optional[float] = None,
                  min_volume: Optional[int] = None,
                  sector: Optional[str] = None) -> List[str]:
    """Filter NASDAQ symbols based on criteria."""
    try:
        data_dir = Path('data')
        screener_files = list(data_dir.glob('nasdaq_screener_*.csv'))
        
        if not screener_files:
            return []
            
        latest_file = max(screener_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_file)
        
        # Apply filters
        if min_market_cap is not None:
            df['Market Cap'] = pd.to_numeric(df['Market Cap'].str.replace('$', '').str.replace(',', ''), errors='coerce')
            df = df[df['Market Cap'] >= min_market_cap]
            
        if min_volume is not None:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df = df[df['Volume'] >= min_volume]
            
        if sector is not None:
            df = df[df['Sector'] == sector]
            
        return df['Symbol'].tolist()
        
    except Exception as e:
        logger.error(f"Error filtering symbols: {e}")
        return [] 
