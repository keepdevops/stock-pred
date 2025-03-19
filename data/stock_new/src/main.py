import sys
import logging
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import polars as pl
import json

# Adjust Python path to include src directory
sys.path.append(str(Path(__file__).parent.parent))

from src.config.config_manager import ConfigurationManager
from src.modules.gui import StockGUI
from src.modules.database import DatabaseConnector
from src.modules.data_loader import DataLoader
from src.modules.stock_ai_agent import StockAIAgent
from src.modules.trading.real_trading_agent import RealTradingAgent
from data_collection_module import DataCollector
from data_cleaning_module import DataCleaner
from database_conversion_module import DatabaseConverter
from ticker_mixing_module import TickerMixer
from gui_module import DataCollectorGUI
from normalization_module import TickerNormalizer

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/market_data.log'),
            logging.StreamHandler()
        ]
    )

def find_databases(base_path: Path = Path.cwd() / "data") -> list[Path]:
    """Find all .duckdb files in the data directory."""
    return list(base_path.glob("**/*.duckdb"))

def initialize_components(root: tk.Tk) -> StockGUI:
    """Initialize all system components and return GUI instance."""
    db_connector = DatabaseConnector()
    data_adapter = DataLoader()
    ai_agent = StockAIAgent(data_adapter)
    return StockGUI(root, db_connector, data_adapter, ai_agent)

def initialize_system():
    """Initialize system components"""
    # Create necessary directories
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/clean").mkdir(parents=True, exist_ok=True)
    Path("data/database").mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    collector = DataCollector()
    cleaner = DataCleaner()
    converter = DatabaseConverter()
    mixer = TickerMixer()
    
    return collector, cleaner, converter, mixer

def analyze_normalized_data(ticker: str, start_date: str, end_date: str):
    # Initialize components
    mixer = TickerMixer()
    normalizer = TickerNormalizer()
    
    # Get ticker data
    data = mixer.execute_combination({
        "name": "single_ticker",
        "tickers": [ticker],
        "fields": ["close", "volume"],
        "date_range": {"start": start_date, "end": end_date}
    })
    
    # Apply normalizations
    normalized_data = normalizer.normalize_ticker_data(
        df=data,
        columns=["close", "volume"],
        methods=["min_max", "z_score"]
    )
    
    # Save metadata
    normalizer.save_normalization_metadata(
        ticker=ticker,
        data_source="yahoo",
        start_date=start_date,
        end_date=end_date
    )
    
    return normalized_data

def analyze_with_advanced_normalization(ticker: str, start_date: str, end_date: str):
    mixer = TickerMixer()
    normalizer = TickerNormalizer()
    
    # Get data
    data = mixer.execute_combination({
        "name": "single_ticker",
        "tickers": [ticker],
        "fields": ["close", "volume"],
        "date_range": {"start": start_date, "end": end_date}
    })
    
    # Apply various normalizations
    results = {}
    
    # Rolling window normalization
    results["rolling"] = normalizer.rolling_normalize(
        df=data,
        column="close",
        window_size=20,
        method="z_score"
    )
    
    # Cross-sectional normalization
    results["cross_sectional"] = normalizer.cross_sectional_normalize(
        df=data,
        columns=["close", "volume"],
        method="min_max"
    )
    
    # Adaptive normalization
    results["adaptive"] = normalizer.adaptive_normalize(
        df=data,
        column="close",
        volatility_window=20
    )
    
    # Compute statistics
    for name, normalized_data in results.items():
        stats = normalizer.compute_normalization_stats(
            data["close"].to_numpy(),
            normalized_data[f"close_{name}_norm"].to_numpy()
        )
        print(f"\nStats for {name} normalization:")
        print(json.dumps(stats, indent=2))
    
    return results

def main():
    # Setup logging
    setup_logging()
    
    try:
        # Create and run GUI
        root = tk.Tk()
        app = DataCollectorGUI(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Error in main application: {e}")
    finally:
        # Ensure proper cleanup
        try:
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    main() 
