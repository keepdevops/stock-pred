"""
Model for training results
"""
class TrainingModel:
    def __init__(self, ticker):
        self.ticker = ticker
        self.model = None
        self.history = None
        self.performance = None
        self.scaler = None
        self.X_test = None
        self.y_test = None
        
    def update_results(self, model, history, performance, scaler=None, X_test=None, y_test=None):
        """Update the training results"""
        self.model = model
        self.history = history
        self.performance = performance
        self.scaler = scaler
        self.X_test = X_test
        self.y_test = y_test
        
    def has_results(self):
        """Check if the model has training results"""
        return self.model is not None and self.history is not None 