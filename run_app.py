import sys
import os
import logging
from pathlib import Path
import tkinter as tk
from tkinter import messagebox

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stock_market_analyzer.config.config_manager import ConfigurationManager
from stock_market_analyzer.modules.gui import StockGUI
from stock_market_analyzer.modules.database import DatabaseConnector
from stock_market_analyzer.modules.data_loader import DataLoader
from stock_market_analyzer.modules.stock_ai_agent import StockAIAgent
from stock_market_analyzer.modules.trading.real_trading_agent import RealTradingAgent

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
        root = tk.Tk()
        app = StockGUI(root, db, data_loader, ai_agent, trading_agent)
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 