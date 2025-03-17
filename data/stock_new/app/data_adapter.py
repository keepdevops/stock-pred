import numpy as np

class DataAdapter:
    def __init__(self):
        self.sequence_length = 10
    
    def prepare_sample_data(self):
        """Create sample data for testing"""
        dates = np.arange(100)
        prices = np.sin(dates / 10) * 100 + 500  # Sample stock prices
        return dates, prices 