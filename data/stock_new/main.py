import sys
import logging
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

# Add the project root directory to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Create necessary directories
log_dir = project_root / "logs"
data_dir = project_root / "data"
log_dir.mkdir(exist_ok=True)
data_dir.mkdir(exist_ok=True)

from modules.gui import StockGUI
from modules.database import DatabaseConnector
from modules.data_loader import DataLoader
from modules.stock_ai_agent import StockAIAgent
from modules.trading.real_trading_agent import RealTradingAgent
from config.config_manager import ConfigurationManager

class StockMarketAnalyzer:
    """Main application class integrating all components."""
    
    def __init__(self, config_path: str = "config/data_collection.json"):
        self.setup_logging()
        self.logger = logging.getLogger("StockMarketAnalyzer")
        
        # Initialize configuration
        self.config_manager = ConfigurationManager(config_path)
        
        # Initialize components
        self.setup_components()
    
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
            # Database setup - Access attributes directly from the dataclass
            db_config = self.config_manager.data_processing.database
            self.db = DatabaseConnector(
                db_path=db_config.path,  # Changed from db_config["path"]
                logger=logging.getLogger("Database")
            )
            
            # Initialize other components
            self.data_loader = DataLoader(
                db_connector=self.db,
                config=self.config_manager.data_collection,
                logger=logging.getLogger("DataLoader")
            )
            
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