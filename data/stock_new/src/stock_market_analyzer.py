"""
Main application class for stock market analysis.
"""
import logging
from .config.data_collection_config import DataCollectionConfig
from .data.data_loader import DataLoader
from .database.database_connector import DatabaseConnector
from .database.nasdaq_database import NasdaqDatabase
from .ai.stock_ai_agent import AIAgent

class StockMarketAnalyzer:
    def __init__(self, market_db: DatabaseConnector, nasdaq_db: NasdaqDatabase):
        """
        Initialize the stock market analyzer.
        
        Args:
            market_db: Market data database connector
            nasdaq_db: NASDAQ symbols database connector
        """
        self.logger = logging.getLogger("StockMarketAnalyzer")
        
        # Store database connections
        self.market_db = market_db
        self.nasdaq_db = nasdaq_db
        
        # Initialize configuration
        self.data_config = DataCollectionConfig()
        
        # Initialize components
        self.data_loader = DataLoader(self.data_config)
        self.ai_agent = AIAgent(self.data_config)
        
        self.logger.info("All components initialized successfully")

    def analyze_stock(self, symbol: str):
        """Analyze a stock symbol."""
        try:
            # Verify symbol exists in NASDAQ database
            symbol_info = self.nasdaq_db.get_symbol_info(symbol)
            if not symbol_info:
                self.logger.warning(f"Symbol {symbol} not found in NASDAQ database")
                return None

            # Collect historical data
            df = self.data_loader.collect_historical_data(symbol)
            if df is None:
                return None

            # Save to market database
            self.market_db.save_stock_data(df, symbol)

            # Train AI model
            self.ai_agent.train(symbol, df)

            # Make predictions
            predictions = self.ai_agent.predict(symbol)

            return predictions

        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return None

    def run(self):
        """Run the analyzer."""
        self.logger.info("Starting application")
        # Add your main application logic here 