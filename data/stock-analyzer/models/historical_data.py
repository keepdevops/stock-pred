"""
Model for historical stock data
"""
import pandas as pd

class HistoricalData:
    def __init__(self, ticker, dataframe=None):
        self.ticker = ticker
        self.dataframe = dataframe if dataframe is not None else pd.DataFrame()
        self.column_mapping = {}
        
    def update_data(self, dataframe):
        """Update the data with a new dataframe"""
        if dataframe is not None and not dataframe.empty:
            self.dataframe = dataframe.copy()
            self._update_column_mapping()
            return True
        return False
        
    def _update_column_mapping(self):
        """Map standard column names to actual column names in the dataframe"""
        mapping = {}
        columns = self.dataframe.columns
        
        # Map date column
        date_cols = [col for col in columns if 'date' in col.lower()]
        if date_cols:
            mapping['Date'] = date_cols[0]
            
        # Map price columns
        price_mappings = {
            'Open': ['open', 'Open', 'open_price', 'Open Price'],
            'High': ['high', 'High', 'high_price', 'High Price'],
            'Low': ['low', 'Low', 'low_price', 'Low Price'],
            'Close': ['close', 'Close', 'close_price', 'Close Price'],
            'Adj Close': ['adj close', 'Adj Close', 'adjusted_close', 'Adjusted Close']
        }
        
        for std_name, possible_names in price_mappings.items():
            for name in possible_names:
                if name in columns:
                    mapping[std_name] = name
                    break
                    
        # Map volume column
        volume_cols = [col for col in columns if 'volume' in col.lower()]
        if volume_cols:
            mapping['Volume'] = volume_cols[0]
            
        # Map sentiment columns
        sentiment_mappings = {
            'Sentiment': ['sentiment', 'Sentiment', 'sentiment_score'],
            'Positive': ['positive', 'Positive', 'positive_score'],
            'Negative': ['negative', 'Negative', 'negative_score'],
            'Neutral': ['neutral', 'Neutral', 'neutral_score']
        }
        
        for std_name, possible_names in sentiment_mappings.items():
            for name in possible_names:
                if name in columns:
                    mapping[std_name] = name
                    break
                    
        self.column_mapping = mapping
        
    def has_data(self):
        """Check if the model has valid data"""
        return not self.dataframe.empty
        
    def get_column(self, std_name):
        """Get a column by its standard name"""
        if std_name in self.column_mapping and self.column_mapping[std_name] in self.dataframe.columns:
            return self.dataframe[self.column_mapping[std_name]]
        return None 