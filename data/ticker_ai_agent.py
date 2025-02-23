import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

class TickerAIAgent:
    def __init__(self, tickers, fields, model_type='lstm', parameters=None, connection=None):
        """Initialize the AI agent with tickers and parameters"""
        self.tickers = tickers if isinstance(tickers, list) else [tickers]
        self.fields = fields if isinstance(fields, list) else [fields]
        self.model_type = model_type.lower()
        self.parameters = parameters or {
            'units': 50,
            'dropout': 0.2,
            'learning_rate': 0.001,
            'epochs': 100
        }
        self.conn = connection
        print("AI agent using database connection")

    def prepare_data(self, ticker, field):
        """Prepare data for training"""
        try:
            print(f"\nPreparing data for {ticker} using {field}")
            
            # Query data from database
            query = f"""
                SELECT date, {field}
                FROM balance_sheets
                WHERE symbol = ?
                ORDER BY date
            """
            df = pd.read_sql_query(query, self.conn, params=(ticker,))
            
            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Set date range for prediction
            last_date = df['date'].max()
            future_date = last_date + timedelta(days=365)
            print(f"Date range: {last_date} to {future_date}")
            print(f"Number of records: {len(df)}")
            
            # Prepare features and target
            values = df[field].values.reshape(-1, 1)
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(values)
            
            # Create sequences
            X, y = [], []
            sequence_length = 10
            
            for i in range(len(scaled_values) - sequence_length):
                X.append(scaled_values[i:i + sequence_length])
                y.append(scaled_values[i + sequence_length])
            
            X = np.array(X)
            y = np.array(y)
            
            # Split into train and validation sets
            train_size = int(len(X) * 0.8)
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:]
            y_val = y[train_size:]
            
            return X_train, y_train, X_val, y_val, scaler
            
        except Exception as e:
            print(f"Data preparation error: {str(e)}")
            raise

    def create_lstm_model(self, input_shape):
        """Create LSTM model"""
        model = Sequential([
            LSTM(units=self.parameters['units'], 
                 input_shape=input_shape,
                 return_sequences=False),
            Dropout(self.parameters['dropout']),
            Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.parameters['learning_rate']),
            loss='mse'
        )
        
        return model

    def train_model(self, ticker, field):
        """Train model for a specific ticker and field"""
        try:
            print(f"\nTraining {self.model_type} model for {ticker} using {field}")
            
            # Prepare data
            X_train, y_train, X_val, y_val, scaler = self.prepare_data(ticker, field)
            
            # Create and train model
            model = self.create_lstm_model(input_shape=(X_train.shape[1], 1))
            
            history = model.fit(
                X_train, y_train,
                epochs=self.parameters['epochs'],
                validation_data=(X_val, y_val),
                verbose=1
            )
            
            # Save model
            os.makedirs('models', exist_ok=True)
            model_path = f"models/{ticker}_{field}_{self.model_type}_model.keras"
            model.save(model_path)
            print(f"\nModel saved: {model_path}")
            
            return model, scaler
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            print(f"Detailed error: {str(e)}")
            raise

    def predict(self, ticker, field, model, scaler):
        """Make predictions for a ticker"""
        try:
            print(f"\nPreparing data for {ticker} using {field}")
            
            # Query recent data
            query = f"""
                SELECT date, {field}
                FROM balance_sheets
                WHERE symbol = ?
                ORDER BY date DESC
                LIMIT 10
            """
            print(f"Executing query with symbol={ticker}")
            df = pd.read_sql_query(query, self.conn, params=(ticker,))
            
            # Prepare input sequence
            values = df[field].values.reshape(-1, 1)
            scaled_values = scaler.transform(values)
            X_pred = scaled_values.reshape(1, -1, 1)
            
            # Make prediction
            scaled_prediction = model.predict(X_pred)
            prediction = scaler.inverse_transform(scaled_prediction)
            
            return prediction[0][0]
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise

    def train_and_predict(self):
        """Train models and make predictions for all tickers and fields"""
        predictions = {}
        
        for ticker in self.tickers:
            predictions[ticker] = {}
            
            for field in self.fields:
                try:
                    # Train model
                    model, scaler = self.train_model(ticker, field)
                    
                    # Make prediction
                    prediction = self.predict(ticker, field, model, scaler)
                    predictions[ticker][field] = prediction
                    
                except Exception as e:
                    print(f"Error processing {ticker} - {field}: {str(e)}")
                    continue
        
        return predictions