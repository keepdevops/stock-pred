import sys
import os
import logging
from pathlib import Path
from PyQt5.QtWidgets import QApplication

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import modules using absolute imports
from stock_market_analyzer.modules.database import DatabaseConnector
from stock_market_analyzer.modules.data_loader import DataLoader
from stock_market_analyzer.modules.stock_ai_agent import StockAIAgent
from stock_market_analyzer.modules.trading.real_trading_agent import RealTradingAgent
from stock_market_analyzer.modules.gui import StockGUI
from stock_market_analyzer.config.config_manager import ConfigurationManager

def setup_logging():
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "stock_analyzer.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Initialize and run the Stock Market Analyzer application."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize configuration manager
        config_manager = ConfigurationManager()
        logger.info("Configuration manager initialized")
        
        # Create database connection with configuration
        db = DatabaseConnector(config_manager.get_database_config())
        logger.info("Database connection established")
        
        # Initialize data loader with configuration
        data_loader = DataLoader(config_manager.get_data_config())
        logger.info("Data loader initialized")
        
        # Initialize AI agent with configuration
        ai_agent = StockAIAgent(config_manager.get_ai_config())
        logger.info("AI agent initialized")
        
        # Initialize trading agent with configuration
        trading_agent = RealTradingAgent(config_manager.get_trading_config())
        logger.info("Trading agent initialized")
        
        # Initialize QApplication
        app = QApplication(sys.argv)
        
        # Create and show main window
        window = StockGUI(db, data_loader, ai_agent, trading_agent, config_manager)
        window.show()
        logger.info("GUI window displayed")
        
        # Start the event loop
        return app.exec_()
        
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        return 1
    finally:
        if 'db' in locals():
            try:
                db.close()
                logger.info("Database connection closed")
            except Exception as e:
                logger.error(f"Error closing database: {e}")

if __name__ == "__main__":
    sys.exit(main()) 