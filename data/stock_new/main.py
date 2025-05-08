import sys
import os
import logging
from pathlib import Path
from src.config.config_manager import ConfigurationManager
from src.data.database import DatabaseConnector
from src.gui.stock_gui import StockGUI
from PyQt5.QtWidgets import QApplication

def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )

def setup_components(logger):
    """Initialize application components."""
    try:
        # Initialize configuration
        config_manager = ConfigurationManager("config/config.json")
        
        # Get database configuration
        db_config = config_manager.get('data_processing', 'database')
        if not db_config:
            logger.warning("No database configuration found, using defaults")
            db_config = {
                'path': 'data/market_data.duckdb',
                'type': 'duckdb'
            }
        
        # Initialize database
        db = DatabaseConnector(db_config['path'])
        
        return config_manager, db
        
    except Exception as e:
        logger.error(f"Error setting up components: {e}")
        raise

def main():
    """Main application entry point."""
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting application")
    
    try:
        # Create application instance
        app = QApplication(sys.argv)
        
        # Set up components
        config_manager, db = setup_components(logger)
        
        # Create and show GUI
        gui = StockGUI(config_manager=config_manager, db=db)
        gui.show()
        
        # Start event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 