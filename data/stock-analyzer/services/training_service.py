"""
Service for handling training operations
"""
from models.training_model import TrainingModel
from utils.event_bus import event_bus

class TrainingService:
    def __init__(self, data_service):
        self.data_service = data_service
        self.training_models = {}  # ticker -> TrainingModel
        
    def train_model(self, params):
        """Train a model with the given parameters"""
        try:
            # Extract parameters
            db_name = params['db_name']
            table_name = params['table_name']
            tickers = params['tickers']
            model_type = params['model_type']
            sequence_length = params['sequence_length']
            neurons = params['neurons']
            layers = params['layers']
            dropout = params['dropout']
            epochs = params['epochs']
            batch_size = params['batch_size']
            learning_rate = params['learning_rate']
            
            # Publish event to indicate training started
            event_bus.publish("training_started", {
                'tickers': tickers,
                'message': f"Training models for {', '.join(tickers)}..."
            })
            
            # Process each ticker
            for ticker in tickers:
                # Get historical data
                historical_data = self.data_service.get_historical_data(ticker)
                
                if not historical_data or not historical_data.has_data():
                    # Load data if not already loaded
                    if not self.data_service.load_historical_data(db_name, table_name, ticker):
                        # Skip this ticker if no data is available
                        event_bus.publish("training_progress", {
                            'ticker': ticker,
                            'status': 'error',
                            'message': f"No data available for {ticker}"
                        })
                        continue
                        
                    # Get the newly loaded data
                    historical_data = self.data_service.get_historical_data(ticker)
                
                # Create a training model for this ticker
                if ticker not in self.training_models:
                    self.training_models[ticker] = TrainingModel(ticker)
                
                # Prepare data for training
                # In a real implementation, you would:
                # 1. Extract features from historical_data
                # 2. Split into training and testing sets
                # 3. Normalize the data
                # 4. Create sequences for the LSTM
                
                # For this example, I'll use placeholders:
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import LSTM, Dense, Dropout
                import numpy as np
                
                # Placeholder code for model creation and training
                model = Sequential()
                model.add(LSTM(neurons, return_sequences=True if layers > 1 else False, 
                             input_shape=(sequence_length, 1)))
                if dropout > 0:
                    model.add(Dropout(dropout))
                    
                # Add additional layers
                for i in range(1, layers):
                    return_sequences = i < layers - 1
                    model.add(LSTM(neurons, return_sequences=return_sequences))
                    if dropout > 0:
                        model.add(Dropout(dropout))
                
                # Add output layer
                model.add(Dense(1))
                
                # Compile model
                model.compile(optimizer='adam', loss='mse')
                
                # Placeholder training data
                X_train = np.random.random((100, sequence_length, 1))
                y_train = np.random.random(100)
                X_test = np.random.random((20, sequence_length, 1))
                y_test = np.random.random(20)
                
                # Train model
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )
                
                # Evaluate model
                performance = model.evaluate(X_test, y_test, verbose=0)
                
                # Update training model
                self.training_models[ticker].update_results(
                    model, history, performance, 
                    scaler=None,  # Would be the actual scaler
                    X_test=X_test, 
                    y_test=y_test
                )
                
                # Publish progress event
                event_bus.publish("training_progress", {
                    'ticker': ticker,
                    'status': 'completed',
                    'message': f"Training completed for {ticker}"
                })
            
            # Publish completion event
            event_bus.publish("training_completed", {
                'tickers': tickers,
                'models': {t: self.training_models[t] for t in tickers if t in self.training_models}
            })
            
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            # Publish error event
            event_bus.publish("training_error", {
                'message': error_msg
            })
            
            return False
            
    def get_training_model(self, ticker):
        """Get training model for a ticker"""
        if ticker in self.training_models:
            return self.training_models[ticker]
        return None 