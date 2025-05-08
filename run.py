import sys
import logging
from PyQt5.QtWidgets import QApplication
from stock_market_analyzer.modules.gui import StockGUI
from stock_market_analyzer.modules.database import Database
from stock_market_analyzer.modules.data_loader import DataLoader
from stock_market_analyzer.modules.ai_agent import AIAgent
from stock_market_analyzer.modules.trading_agent import TradingAgent
import json

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_config():
    """Load configuration from config.json."""
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return {}

def main():
    """Main entry point for the application."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize components
        db = Database(config.get('database', {}).get('path', 'stocks.db'))
        data_loader = DataLoader(config.get('data', {}))
        ai_agent = AIAgent(config.get('ai', {}))
        trading_agent = TradingAgent(config.get('trading', {}))
        
        # Create Qt application
        app = QApplication(sys.argv)
        
        # Create and show main window
        window = StockGUI(db, data_loader, ai_agent, trading_agent)
        window.show()
        
        # Start event loop
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 