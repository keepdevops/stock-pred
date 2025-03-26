import sys
import logging
from pathlib import Path
from PyQt5.QtWidgets import QApplication

from .config.config_manager import ConfigurationManager
from .modules.gui import StockGUI
from .modules.database import DatabaseConnector
from .modules.data_loader import DataLoader
from .modules.stock_ai_agent import StockAIAgent
from .modules.trading.real_trading_agent import RealTradingAgent

def main():
    """Initialize and run the Stock Market Analyzer application."""
    try:
        # Initialize configuration
        config = ConfigurationManager()
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger(__name__)
        
        # Initialize components
        db = DatabaseConnector(config.get_database_config())
        data_loader = DataLoader(config.get_data_config())
        ai_agent = StockAIAgent(config.get_ai_config())
        trading_agent = RealTradingAgent(config.get_trading_config())
        
        # Create and run GUI
        app = QApplication(sys.argv)
        window = StockGUI(db, data_loader, ai_agent, trading_agent)
        window.show()
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 