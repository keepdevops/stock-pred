"""
Model for prediction results
"""
class PredictionModel:
    def __init__(self, ticker):
        self.ticker = ticker
        self.historical_dates = []
        self.historical_prices = []
        self.prediction_dates = []
        self.predicted_prices = []
        self.upper_bound = []
        self.lower_bound = []
        self.validation_prices = []
        
    def update_predictions(self, historical_dates, historical_prices, 
                          prediction_dates, predicted_prices,
                          upper_bound=None, lower_bound=None,
                          validation_prices=None):
        """Update the prediction results"""
        self.historical_dates = historical_dates
        self.historical_prices = historical_prices
        self.prediction_dates = prediction_dates
        self.predicted_prices = predicted_prices
        self.upper_bound = upper_bound if upper_bound is not None else []
        self.lower_bound = lower_bound if lower_bound is not None else []
        self.validation_prices = validation_prices if validation_prices is not None else []
        
    def has_predictions(self):
        """Check if the model has prediction results"""
        return len(self.predicted_prices) > 0 