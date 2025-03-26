import logging
import sys
from pathlib import Path
from PyQt5.QtWidgets import QApplication

from .modules.database import Database
from .modules.data_loader import DataLoader
from .modules.ai_agent import AIAgent
from .modules.trading_agent import TradingAgent
from .modules.gui import StockGUI

def setup_logging():
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "stock_analyzer.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )

def main():
    """Main entry point for the application."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        app = QApplication(sys.argv)
        
        # Create database connection
        db = Database()
        
        # Initialize data loader
        data_loader = DataLoader({'source': 'yahoo'})
        
        # Initialize AI agent
        ai_agent = AIAgent()
        
        # Initialize trading agent
        trading_agent = TradingAgent()
        
        # Create and show main window
        window = StockGUI(db, data_loader, ai_agent, trading_agent)
        window.show()
        
        # Start the event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
    finally:
        if 'db' in locals():
            db.close()

if __name__ == "__main__":
    main() 