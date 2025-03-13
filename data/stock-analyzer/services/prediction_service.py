"""
Service for handling data operations
"""
from database.database_manager import DatabaseManager
from models.historical_data import HistoricalData
from utils.event_bus import event_bus

class DataService:
    def __init__(self):
        self.database_manager = DatabaseManager()
        self.historical_data = {}  # ticker -> HistoricalData
        
    def get_available_databases(self):
        """Get available databases"""
        return self.database_manager.get_available_databases()
        
    def get_tables(self, db_name):
        """Get tables in a database"""
        return self.database_manager.get_tables(db_name)
        
    def get_tickers(self, db_name, table_name):
        """Get tickers in a table"""
        return self.database_manager.get_tickers(db_name, table_name)
        
    def load_historical_data(self, db_name, table_name, ticker):
        """Load historical data for a ticker"""
        # Get data from database
        df = self.database_manager.get_data(db_name, table_name, ticker=ticker)
        
        # Create or update historical data model
        if ticker not in self.historical_data:
            self.historical_data[ticker] = HistoricalData(ticker)
            
        # Update data and notify subscribers
        if self.historical_data[ticker].update_data(df):
            # Publish event to notify UI components
            event_bus.publish("historical_data_updated", {
                'ticker': ticker,
                'data': self.historical_data[ticker]
            })
            return True
        
        event_bus.publish("historical_data_error", {
            'ticker': ticker,
            'message': f"No data available for {ticker}"
        })
        return False
        
    def get_historical_data(self, ticker):
        """Get historical data for a ticker"""
        if ticker in self.historical_data:
            return self.historical_data[ticker]
        return None 