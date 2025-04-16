import logging
import sys
import os
from pathlib import Path
from PyQt6.QtWidgets import QApplication

# Add the current directory to the Python path
current_dir = str(Path(__file__).resolve().parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from modules.database import DatabaseConnector
from modules.data_loader import DataLoader
from modules.stock_ai_agent import StockAIAgent
from modules.trading.real_trading_agent import RealTradingAgent
from modules.gui import MainWindow

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
    """Main entry point for the application."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize components
        app = QApplication(sys.argv)
        
        # Create database connection
        db = DatabaseConnector()
        
        # Initialize data loader
        data_loader = DataLoader()
        
        # Initialize AI agent
        ai_agent = StockAIAgent()
        
        # Initialize trading agent
        trading_agent = RealTradingAgent()
        
        # Create and show main window
        window = MainWindow(db, data_loader, ai_agent, trading_agent)
        window.show()
        
        # Start the event loop
        sys.exit(app.exec())
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)
    finally:
        if 'db' in locals():
            db.close()

if __name__ == "__main__":
    main() 