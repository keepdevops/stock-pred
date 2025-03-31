import sys
import os
import logging
from pathlib import Path
from PyQt5.QtWidgets import QApplication

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Debug prints
print("Python path:", sys.path)
print("Current directory:", os.getcwd())
print("Project root:", project_root)
print("Modules directory:", project_root / "stock_market_analyzer" / "modules")
print("Database module path:", project_root / "stock_market_analyzer" / "modules" / "database.py")

# Import modules using absolute imports
from stock_market_analyzer.modules.database import DatabaseConnector
from stock_market_analyzer.modules.data_loader import DataLoader
from stock_market_analyzer.modules.stock_ai_agent import StockAIAgent
from stock_market_analyzer.modules.trading.real_trading_agent import RealTradingAgent
from stock_market_analyzer.modules.gui import StockGUI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for the application"""
    try:
        # Initialize database connector
        db_connector = DatabaseConnector()
        
        # Initialize data loader with configuration
        data_loader_config = {
            'source': 'yahoo',
            'start_date': '2020-01-01',
            'end_date': 'today',
            'data_dir': 'data'
        }
        data_loader = DataLoader(data_loader_config)
        
        # Initialize AI agent with configuration
        ai_agent_config = {
            'lookback_days': 60,
            'prediction_days': 5,
            'training': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'validation_split': 0.2
            }
        }
        ai_agent = StockAIAgent(ai_agent_config)
        
        # Initialize trading agent with configuration
        trading_agent_config = {
            'mode': 'simulation',
            'initial_balance': 10000,
            'risk_per_trade': 0.02
        }
        trading_agent = RealTradingAgent(trading_agent_config)
        
        # Initialize and run GUI
        app = QApplication(sys.argv)
        gui = StockGUI(db_connector, data_loader, ai_agent, trading_agent)
        gui.show()
        
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Error in main: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main() 