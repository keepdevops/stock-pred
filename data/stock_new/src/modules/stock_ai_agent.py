from typing import Dict, Any, Optional
import numpy as np
from .models.model_factory import ModelFactory
from .data_adapter import DataAdapter

class StockAIAgent:
    def __init__(
        self,
        data_adapter: DataAdapter,
        model_type: str = "LSTM",
        model_config: Optional[Dict[str, Any]] = None
    ):
        self.data_adapter = data_adapter
        self.model_type = model_type
        self.model_config = model_config or {
            'lstm_units': [50, 50],
            'dense_units': [32],
            'dropout': 0.2,
            'optimizer': 'adam',
            'loss': 'mse'
        }
        self.model = None
    
    def build_model(self, input_shape):
        """Build the neural network model."""
        self.model = ModelFactory.create_model(
            self.model_type,
            input_shape,
            self.model_config
        )
        return self.model
    
    def train(self, X, y, validation_data=None, epochs=100, batch_size=32):
        """Train the model."""
        if self.model is None:
            self.build_model(X.shape[1:])
        
        history = self.model.fit(
            X, y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        return history
    
    def predict(self, X):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        return self.model.predict(X)
    
    def predict_future(self, X, days=30):
        """Predict future values."""
        predictions = []
        current_sequence = X[-1:]  # Start with last known sequence
        
        for _ in range(days):
            pred = self.predict(current_sequence)[0, 0]
            predictions.append(pred)
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = pred
        
        return np.array(predictions) 