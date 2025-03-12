"""
Service for handling prediction operations
"""
from models.prediction_model import PredictionModel
from utils.event_bus import event_bus
import pandas as pd
import numpy as np
import datetime

class PredictionService:
    def __init__(self, data_service, training_service):
        self.data_service = data_service
        self.training_service = training_service
        self.prediction_models = {}  # ticker -> PredictionModel
        
    def make_predictions(self, params):
        """Make predictions with the given parameters"""
        try:
            # Extract parameters
            db_name = params['db_name']
            table_name = params['table_name']
            tickers = params['tickers']
            days = params['days']
            
            # Publish event to indicate prediction started
            event_bus.publish("prediction_started", {
                'tickers': tickers,
                'message': f"Making predictions for {', '.join(tickers)}..."
            })
            
            # Process each ticker
            for ticker in tickers:
                # Get historical data
                historical_data = self.data_service.get_historical_data(ticker)
                
                if not historical_data or not historical_data.has_data():
                    # Load data if not already loaded
                    if not self.data_service.load_historical_data(db_name, table_name, ticker):
                        # Skip this ticker if no data is available
                        event_bus.publish("prediction_progress", {
                            'ticker': ticker,
                            'status': 'error',
                            'message': f"No data available for {ticker}"
                        })
                        continue
                        
                    # Get the newly loaded data
                    historical_data = self.data_service.get_historical_data(ticker)
                
                # Create a prediction model for this ticker
                if ticker not in self.prediction_models:
                    self.prediction_models[ticker] = PredictionModel(ticker)
                
                # Get training model
                training_model = self.training_service.get_training_model(ticker)
                
                # Prepare data for prediction
                df = historical_data.dataframe
                
                # Get dates and prices
                date_col = historical_data.column_mapping.get('Date')
                close_col = historical_data.column_mapping.get('Close')
                
                if not date_col or not close_col:
                    event_bus.publish("prediction_progress", {
                        'ticker': ticker,
                        'status': 'error',
                        'message': f"Required columns not found for {ticker}"
                    })
                    continue
                
                # Convert dates to datetime
                dates = pd.to_datetime(df[date_col])
                prices = df[close_col].values
                
                # Generate future dates
                last_date = dates.iloc[-1]
                future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, days+1)]
                
                # Generate predictions (placeholder)
                # In a real implementation, you would use the trained model to make predictions
                if training_model and training_model.has_results():
                    # Simulated predictions using the trained model
                    last_price = prices[-1]
                    # Adding some random variations to simulate predictions
                    predicted_prices = [last_price * (1 + np.random.normal(0.001, 0.01)) for _ in range(days)]
                    for i in range(1, days):
                        predicted_prices[i] = predicted_prices[i-1] * (1 + np.random.normal(0.001, 0.01))
                    
                    # Generate confidence intervals
                    upper_bound = [p * (1 + 0.05) for p in predicted_prices]
                    lower_bound = [p * (1 - 0.05) for p in predicted_prices]
                else:
                    # Fallback to simple moving average prediction
                    window = min(10, len(prices))
                    avg_pct_change = np.mean(np.diff(prices[-window:]) / prices[-window:-1])
                    
                    predicted_prices = [prices[-1]]
                    for i in range(1, days):
                        predicted_prices.append(predicted_prices[-1] * (1 + avg_pct_change))
                    
                    predicted_prices = predicted_prices[1:]  # Remove the first element (last actual price)
                    
                    # Simple confidence bounds
                    upper_bound = [p * 1.05 for p in predicted_prices]
                    lower_bound = [p * 0.95 for p in predicted_prices]
                
                # Update prediction model
                self.prediction_models[ticker].update_predictions(
                    historical_dates=dates.tolist(),
                    historical_prices=prices.tolist(),
                    prediction_dates=future_dates,
                    predicted_prices=predicted_prices,
                    upper_bound=upper_bound,
                    lower_bound=lower_bound
                )
                
                # Publish progress event
                event_bus.publish("prediction_progress", {
                    'ticker': ticker,
                    'status': 'completed',
                    'message': f"Prediction completed for {ticker}"
                })
            
            # Publish completion event
            event_bus.publish("prediction_completed", {
                'tickers': tickers,
                'models': {t: self.prediction_models[t] for t in tickers if t in self.prediction_models}
            })
            
            return True
            
        except Exception as e:
            import traceback
            error_msg = f"Prediction error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            # Publish error event
            event_bus.publish("prediction_error", {
                'message': error_msg
            })
            
            return False
            
    def get_prediction_model(self, ticker):
        """Get prediction model for a ticker"""
        if ticker in self.prediction_models:
            return self.prediction_models[ticker]
        return None 