import sys
import logging
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
import json

# Add the project root directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Create necessary directories
log_dir = project_root / "logs"
data_dir = project_root / "data"
config_dir = project_root / "config"
log_dir.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)
config_dir.mkdir(exist_ok=True)

from modules.gui import StockGUI
from modules.database import DatabaseConnector
from modules.data_loader import DataLoader
from modules.stock_ai_agent import StockAIAgent
from modules.trading.real_trading_agent import RealTradingAgent
from config.config_manager import ConfigurationManager

class StockMarketAnalyzer:
    """Main application class integrating all components."""
    
    def __init__(self, config_path: str = "config/system_config.json"):
        self.setup_logging()
        self.logger = logging.getLogger("StockMarketAnalyzer")
        
        # Initialize configuration
        try:
            self.config_manager = ConfigurationManager(config_path)
        except FileNotFoundError:
            # Create default config file if it doesn't exist
            self.create_default_config(config_path)
            self.config_manager = ConfigurationManager(config_path)
        
        # Initialize components
        self.setup_components()
    
    def create_default_config(self, config_path: str) -> None:
        """Create default configuration file."""
        default_config = {
            "system_config": {
                "version": "1.0.0",
                "description": "Configuration for market data collection, processing, and analysis system",
                "date": "2025-03-17"
            },
            "data_collection": {
                "tickers": ["AAPL", "GOOG", "MSFT"],
                "historical": {
                    "enabled": True,
                    "points": 720,
                    "start_date": "2023-03-17",
                    "end_date": "2025-03-17"
                },
                "realtime": {
                    "enabled": False,
                    "source": "yahoo",
                    "available_sources": [
                        {
                            "name": "yahoo",
                            "type": "http",
                            "retry_attempts": 3,
                            "retry_backoff_base": 2
                        }
                    ]
                },
                "parallel_processing": {
                    "enabled": True,
                    "max_workers": 10
                }
            },
            "cache_settings": {
                "enabled": True,
                "database": "sqlite",
                "path": "cache/realtime_cache.db",
                "expiry_seconds": 300,
                "max_entries": 1000
            },
            "data_processing": {
                "cleaning": {
                    "lowercase": True,
                    "remove_special_chars": True,
                    "standardize_dates": "YYYY-MM-DD",
                    "fill_missing": "0"
                },
                "validation": {
                    "enabled": True,
                    "batch_size": 10,
                    "required_columns": [
                        "date", "open", "high", "low", "close", "volume"
                    ],
                    "date_format": "YYYY-MM-DD",
                    "numeric_fields": [
                        "open", "high", "low", "close", "volume"
                    ]
                },
                "database": {
                    "type": "duckdb",
                    "path": "data/market_data.duckdb",
                    "index_columns": ["date"]
                }
            },
            "ticker_mixing": {
                "combinations": [
                    {
                        "name": "tech_portfolio",
                        "tickers": ["AAPL", "GOOG"],
                        "fields": ["date", "close"],
                        "filters": {"date": "> '2024-01-01'"},
                        "aggregations": {"close": "AVG"}
                    }
                ],
                "output_format": "csv",
                "output_path": "data/"
            },
            "gui_settings": {
                "theme": "clam",
                "available_themes": ["clam", "alt", "default"],
                "window_size": "800x600"
            },
            "logging": {
                "enabled": True,
                "files": [
                    {
                        "name": "data_collection",
                        "path": "logs/data_collection.log",
                        "level": "INFO"
                    },
                    {
                        "name": "db_conversion",
                        "path": "logs/db_conversion.log",
                        "level": "INFO"
                    }
                ]
            }
        }
        
        config_file = Path(config_path)
        config_file.parent.mkdir(exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        
        self.logger.info(f"Created default configuration file: {config_path}")
    
    def setup_logging(self) -> None:
        """Configure application logging."""
        try:
            # Create logs directory if it doesn't exist
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            
            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(log_dir / 'app.log')
                ]
            )
            
        except Exception as e:
            print(f"Error setting up logging: {str(e)}")
            sys.exit(1)
    
    def setup_components(self) -> None:
        """Initialize all application components."""
        try:
            # Database setup
            db_config = self.config_manager.data_processing.database
            self.db = DatabaseConnector(config={"path": db_config.path})
            
            # Initialize other components
            self.data_loader = DataLoader(config=self.config_manager.data_collection)
            
            self.ai_agent = StockAIAgent(
                db_connector=self.db,
                logger=logging.getLogger("AIAgent")
            )
            
            self.trading_agent = RealTradingAgent(
                db_connector=self.db,
                logger=logging.getLogger("TradingAgent")
            )
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise
    
    def run(self) -> None:
        """Start the application."""
        try:
            # Create main window
            root = tk.Tk()
            root.title("Stock Market Analyzer")
            
            # Apply GUI settings from config
            if hasattr(self.config_manager, 'gui_settings'):
                root.geometry(self.config_manager.gui_settings.window_size)
                root.tk.call("ttk::style", "theme", "use", self.config_manager.gui_settings.theme)
            
            # Initialize GUI with all components
            self.gui = StockGUI(
                root,
                self.db,
                self.data_loader,
                self.ai_agent,
                self.trading_agent,
                self.config_manager
            )
            
            # Start the application
            self.logger.info("Starting application")
            root.mainloop()
            
        except Exception as e:
            self.logger.error(f"Error running application: {str(e)}")
            messagebox.showerror("Error", f"Application error: {str(e)}")
            sys.exit(1)
    
    def cleanup(self) -> None:
        """Cleanup resources before exit."""
        try:
            self.db.close()
            self.logger.info("Application cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

def main():
    """Application entry point."""
    try:
        app = StockMarketAnalyzer()
        app.run()
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        messagebox.showerror("Fatal Error", str(e))
        sys.exit(1)
    finally:
        if 'app' in locals():
            app.cleanup()

if __name__ == "__main__":
    main() 