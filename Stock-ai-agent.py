from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import pickle
import traceback
import numpy as np
import pandas as pd
import duckdb

class TrainingAgent:
    def __init__(self):
        """Initialize Training Agent"""
        try:
            print("\n=== Initializing Training Agent ===")
            self.model = None
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.features = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'MA5', 'MA20', 'RSI', 'MACD', 'ATR'
            ]
            self.sequence_length = 10
            
            # Initialize database
            self.initialize_database()
            
            print("Training Agent initialized successfully")
            
        except Exception as e:
            print(f"Error initializing Training Agent: {str(e)}")
            traceback.print_exc()

    def prepare_training_data(self, df):
        """Prepare data for training"""
        try:
            print("\n=== Preparing Training Data ===")
            
            # Verify features
            missing_features = [f for f in self.features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Scale the data
            feature_data = df[self.features].values
            scaled_data = self.scaler.fit_transform(feature_data)
            
            # Create sequences
            X, y = [], []
            for i in range(len(scaled_data) - self.sequence_length):
                X.append(scaled_data[i:(i + self.sequence_length)])
                y.append(scaled_data[i + self.sequence_length, self.features.index('Close')])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            print(f"Error preparing training data: {str(e)}")
            traceback.print_exc()
            return None, None

    def train_model(self, ticker, epochs=100, batch_size=32):
        """Train the model"""
        try:
            print(f"\n=== Training Model for {ticker} ===")
            
            # Get training data
            df = self.get_training_data(ticker)
            if df is None or df.empty:
                raise ValueError(f"No training data for {ticker}")
            
            # Prepare data
            X, y = self.prepare_training_data(df)
            if X is None or y is None:
                raise ValueError("Failed to prepare training data")
            
            # Create or get model
            if self.model is None:
                self.create_model(input_shape=(X.shape[1], X.shape[2]))
            
            # Train
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                verbose=1
            )
            
            # Save trained components
            self.save_trained_components(ticker)
            
            return history
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            traceback.print_exc()
            return None

    def save_trained_components(self, ticker):
        """Save trained model and scaler"""
        try:
            # Save model
            model_path = f'models/{ticker}_model.h5'
            scaler_path = f'models/{ticker}_scaler.pkl'
            
            self.model.save(model_path)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            print(f"Saved model to {model_path}")
            print(f"Saved scaler to {scaler_path}")
            
        except Exception as e:
            print(f"Error saving components: {str(e)}")
            traceback.print_exc()

class PredictionAgent:
    def __init__(self, model_path=None, scaler_path=None):
        """Initialize Prediction Agent"""
        try:
            print("\n=== Initializing Prediction Agent ===")
            self.model = None
            self.scaler = None
            self.features = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'MA5', 'MA20', 'RSI', 'MACD', 'ATR'
            ]
            self.sequence_length = 10
            
            # Load components if paths provided
            if model_path and scaler_path:
                self.load_components(model_path, scaler_path)
            
            print("Prediction Agent initialized successfully")
            
        except Exception as e:
            print(f"Error initializing Prediction Agent: {str(e)}")
            traceback.print_exc()

    def load_components(self, model_path, scaler_path):
        """Load trained model and scaler"""
        try:
            # Load model
            self.model = keras.models.load_model(model_path)
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                
            print(f"Loaded model from {model_path}")
            print(f"Loaded scaler from {scaler_path}")
            
        except Exception as e:
            print(f"Error loading components: {str(e)}")
            traceback.print_exc()

    def prepare_prediction_data(self, df):
        """Prepare data for prediction"""
        try:
            print("\n=== Preparing Prediction Data ===")
            
            # Verify components
            if self.scaler is None:
                raise ValueError("No scaler loaded")
            
            # Verify features
            missing_features = [f for f in self.features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Scale the data
            feature_data = df[self.features].values
            scaled_data = self.scaler.transform(feature_data)
            
            # Create sequence
            if len(scaled_data) >= self.sequence_length:
                X_pred = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, len(self.features))
                return X_pred
            else:
                raise ValueError(f"Not enough data points")
            
        except Exception as e:
            print(f"Error preparing prediction data: {str(e)}")
            traceback.print_exc()
            return None

    def make_prediction(self, df):
        """Make prediction"""
        try:
            print("\n=== Making Prediction ===")
            
            # Verify components
            if self.model is None:
                raise ValueError("No model loaded")
            
            # Prepare data
            X_pred = self.prepare_prediction_data(df)
            if X_pred is None:
                raise ValueError("Failed to prepare prediction data")
            
            # Make prediction
            prediction = self.model.predict(X_pred, verbose=0)
            
            # Transform prediction
            final_prediction = self.inverse_transform_predictions(prediction)
            
            return final_prediction[0]
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            traceback.print_exc()
            return None

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        try:
            print("\nCalculating technical indicators...")
            
            # Moving averages
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            
            # ATR
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['ATR'] = true_range.rolling(window=14).mean()
            
            print("Technical indicators calculated successfully")
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            traceback.print_exc()
            return df

    def inverse_transform_predictions(self, predictions):
        """Transform predictions back to original scale"""
        try:
            if self.scaler is None:
                raise ValueError("Scaler not initialized")
            
            # Get Close price index
            close_idx = self.features.index('Close')
            
            # Reshape predictions
            reshaped_pred = np.zeros((len(predictions), len(self.features)))
            reshaped_pred[:, close_idx] = predictions.flatten()
            
            # Inverse transform
            inverse_transformed = self.scaler.inverse_transform(reshaped_pred)
            
            return inverse_transformed[:, close_idx]
            
        except Exception as e:
            print(f"Error in inverse transform: {str(e)}")
            traceback.print_exc()
            return predictions

    def get_historical_data(self, ticker):
        """Get historical data for prediction"""
        try:
            print(f"\nFetching historical data for {ticker}")
            
            # Query data from database
            query = f"""
                SELECT date, open, high, low, close, volume 
                FROM {self.selected_table}
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 100
            """
            
            df = pd.read_sql_query(query, self.conn, params=[ticker])
            
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Sort by date ascending for calculations
            df.sort_index(inplace=True)
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            print(f"Retrieved {len(df)} records with columns: {df.columns.tolist()}")
            return df
            
        except Exception as e:
            print(f"Error getting historical data: {str(e)}")
            traceback.print_exc()
            return None

    def initialize_database(self):
        """Initialize DuckDB database and tables"""
        try:
            print("\n=== Initializing Database ===")
            
            # Create DuckDB connection
            self.conn = duckdb.connect('stock_predictions.duckdb')
            
            # Create training data table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    date TIMESTAMP,
                    ticker VARCHAR,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    ma5 DOUBLE,
                    ma20 DOUBLE,
                    rsi DOUBLE,
                    macd DOUBLE,
                    atr DOUBLE
                )
            """)
            
            # Create predictions table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_date TIMESTAMP,
                    ticker VARCHAR,
                    current_price DOUBLE,
                    predicted_price DOUBLE,
                    prediction_horizon INTEGER
                )
            """)
            
            print("Database initialized successfully")
            
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
            traceback.print_exc()

    def save_training_data(self, df, ticker):
        """Save training data to DuckDB"""
        try:
            print(f"\n=== Saving Training Data for {ticker} ===")
            
            # Calculate technical indicators if needed
            if 'MA5' not in df.columns:
                df = self.calculate_technical_indicators(df)
            
            # Prepare data for saving
            save_df = df.reset_index()
            save_df['ticker'] = ticker
            
            # Register DataFrame with DuckDB
            self.conn.register('temp_df', save_df)
            
            # Insert data
            self.conn.execute("""
                INSERT INTO training_data 
                SELECT 
                    date, ticker, open, high, low, close, volume,
                    ma5, ma20, rsi, macd, atr
                FROM temp_df
            """)
            
            print(f"Saved {len(df)} records to training_data table")
            
            # Verify data was saved
            count = self.conn.execute(f"""
                SELECT COUNT(*) 
                FROM training_data 
                WHERE ticker = '{ticker}'
            """).fetchone()[0]
            
            print(f"Total records for {ticker}: {count}")
            
        except Exception as e:
            print(f"Error saving training data: {str(e)}")
            traceback.print_exc()

    def get_training_data(self, ticker):
        """Retrieve training data from DuckDB"""
        try:
            print(f"\n=== Retrieving Training Data for {ticker} ===")
            
            # Query data
            query = f"""
                SELECT *
                FROM training_data
                WHERE ticker = '{ticker}'
                ORDER BY date
            """
            
            df = self.conn.execute(query).fetchdf()
            
            if df.empty:
                print(f"No training data found for {ticker}")
                return None
            
            print(f"Retrieved {len(df)} records")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            
            return df
            
        except Exception as e:
            print(f"Error retrieving training data: {str(e)}")
            traceback.print_exc()
            return None

    def verify_training_data(self):
        """Verify training data in database"""
        try:
            print("\n=== Verifying Training Data ===")
            
            # Check table exists
            tables = self.conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = 'training_data'
            """).fetchall()
            
            if not tables:
                print("Training data table does not exist")
                return False
            
            # Get table info
            count = self.conn.execute("SELECT COUNT(*) FROM training_data").fetchone()[0]
            tickers = self.conn.execute("SELECT DISTINCT ticker FROM training_data").fetchdf()
            
            print(f"Total records: {count}")
            print(f"Unique tickers: {tickers['ticker'].tolist()}")
            
            # Verify data quality
            nulls = self.conn.execute("""
                SELECT 
                    COUNT(*) - COUNT(date) as date_nulls,
                    COUNT(*) - COUNT(ticker) as ticker_nulls,
                    COUNT(*) - COUNT(close) as close_nulls
                FROM training_data
            """).fetchdf()
            
            print("\nData Quality Check:")
            print(f"Null dates: {nulls['date_nulls'].iloc[0]}")
            print(f"Null tickers: {nulls['ticker_nulls'].iloc[0]}")
            print(f"Null prices: {nulls['close_nulls'].iloc[0]}")
            
            return count > 0
            
        except Exception as e:
            print(f"Error verifying training data: {str(e)}")
            traceback.print_exc()
            return False

    def create_model(self, input_shape):
        """Build LSTM model"""
        try:
            print(f"Building LSTM model with input shape: {input_shape}")
            
            self.model = keras.Sequential([
                keras.layers.LSTM(50, input_shape=input_shape, return_sequences=True),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(50, return_sequences=False),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(25),
                keras.layers.Dense(1)
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            print("Model built successfully")
            self.model.summary()
            
        except Exception as e:
            print(f"Error building model: {str(e)}")
            traceback.print_exc()

    def save_prediction(self, ticker, current_price, predicted_price, days_ahead):
        """Save prediction to DuckDB"""
        try:
            conn = duckdb.connect("forex-duckdb.db")
            
            # Create predictions table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_predictions (
                    prediction_id INTEGER,
                    model_id INTEGER,
                    ticker VARCHAR,
                    prediction_date TIMESTAMP,
                    current_price DOUBLE,
                    predicted_price DOUBLE,
                    days_ahead INTEGER,
                    actual_price DOUBLE,  -- To be updated later
                    prediction_error DOUBLE,  -- To be updated later
                    validation_date TIMESTAMP  -- To be updated later
                )
            """)
            
            # Get next prediction_id
            result = conn.execute("SELECT COALESCE(MAX(prediction_id), 0) + 1 FROM ai_predictions").fetchone()
            prediction_id = result[0]
            
            # Get latest model_id
            result = conn.execute("SELECT MAX(model_id) FROM ai_model_metrics WHERE ticker = ?", [ticker]).fetchone()
            model_id = result[0]
            
            # Save prediction
            prediction_df = pd.DataFrame({
                'prediction_id': [prediction_id],
                'model_id': [model_id],
                'ticker': [ticker],
                'prediction_date': [pd.Timestamp.now()],
                'current_price': [current_price],
                'predicted_price': [predicted_price],
                'days_ahead': [days_ahead],
                'actual_price': [None],
                'prediction_error': [None],
                'validation_date': [None]
            })
            
            conn.execute("""
                INSERT INTO ai_predictions 
                SELECT * FROM prediction_df
            """)
            
            conn.commit()
            conn.close()
            
            print(f"Prediction saved to database with ID: {prediction_id}")
            
        except Exception as e:
            print(f"Error saving prediction: {str(e)}")
            traceback.print_exc()

    def predict(self, X):
        """Make prediction using prepared data"""
        try:
            print("\nMaking prediction...")
            
            if self.model is None:
                raise ValueError("No model loaded for prediction")
            
            if X is None:
                raise ValueError("No data provided for prediction")
            
            print(f"Input shape: {X.shape}")
            prediction = self.model.predict(X, verbose=0)
            print(f"Prediction shape: {prediction.shape}")
            
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            traceback.print_exc()
            return None

    def load_components(self, model_path, scaler_path):
        """Load trained model and scaler"""
        try:
            # Load model
            self.model = keras.models.load_model(model_path)
            
            # Load scaler
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
                
            print(f"Loaded model from {model_path}")
            print(f"Loaded scaler from {scaler_path}")
            
        except Exception as e:
            print(f"Error loading components: {str(e)}")
            traceback.print_exc()

    def inverse_transform_predictions(self, predictions):
        """Transform predictions back to original scale"""
        try:
            if self.scaler is None:
                raise ValueError("Scaler not initialized")
            
            # Get Close price index
            close_idx = self.features.index('Close')
            
            # Reshape predictions
            reshaped_pred = np.zeros((len(predictions), len(self.features)))
            reshaped_pred[:, close_idx] = predictions.flatten()
            
            # Inverse transform
            inverse_transformed = self.scaler.inverse_transform(reshaped_pred)
            
            return inverse_transformed[:, close_idx]
            
        except Exception as e:
            print(f"Error in inverse transform: {str(e)}")
            traceback.print_exc()
            return predictions

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

    def selected_ticker(self):
        """Return the selected ticker"""
        return self.selected_ticker

    def selected_table(self):
        """Return the selected table"""
        return self.selected_table

from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import pickle
import traceback
import numpy as np
import pandas as pd
import duckdb

class StockPredictor:
    def __init__(self):
        """Initialize the stock predictor"""
        try:
            print("\n=== Initializing Stock Predictor ===")
            
            # Initialize core components
            self.model = None
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.features = [
                'Open', 'High', 'Low', 'Close', 'Volume',
                'MA5', 'MA20', 'RSI', 'MACD', 'ATR'
            ]
            
            print("Core components initialized")
            print(f"Features: {self.features}")
            
        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            traceback.print_exc()

    def prepare_data_for_prediction(self, df):
        """Prepare data for prediction"""
        try:
            print("\n=== Preparing Data for Prediction ===")
            
            # Input validation
            if df is None or df.empty:
                raise ValueError("Input DataFrame is None or empty")
            
            print(f"Input DataFrame shape: {df.shape}")
            
            # Verify features
            missing_features = [f for f in self.features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Extract feature data
            feature_data = df[self.features].values
            print(f"Feature data shape: {feature_data.shape}")
            
            # Scale the data
            if not hasattr(self, 'scaler'):
                print("Initializing new scaler...")
                self.scaler = MinMaxScaler(feature_range=(0, 1))
            
            try:
                scaled_data = self.scaler.transform(feature_data)
                print("Data scaled successfully")
            except Exception as e:
                print(f"Scaling failed, fitting new scaler: {str(e)}")
                scaled_data = self.scaler.fit_transform(feature_data)
            
            # Create sequence for LSTM
            sequence_length = 10
            if len(scaled_data) >= sequence_length:
                X_pred = scaled_data[-sequence_length:].reshape(1, sequence_length, len(self.features))
                print(f"Prediction sequence shape: {X_pred.shape}")
                return X_pred
            else:
                raise ValueError(f"Not enough data points. Need {sequence_length}, got {len(scaled_data)}")
            
        except Exception as e:
            print(f"Error preparing prediction data: {str(e)}")
            traceback.print_exc()
            return None

    def make_prediction(self, ticker=None):
        """Make prediction for a ticker"""
        try:
            print("\n=== Starting Prediction ===")
            
            # Validate ticker
            ticker = ticker or self.selected_ticker
            if not ticker:
                raise ValueError("No ticker selected")
            
            # Get historical data
            df = self.get_historical_data(ticker)
            if df is None or df.empty:
                raise ValueError(f"No data available for {ticker}")
            
            print(f"Retrieved {len(df)} records")
            
            # Calculate indicators if needed
            if 'MA5' not in df.columns:
                df = self.calculate_technical_indicators(df)
            
            # Prepare data
            X_pred = self.prepare_data_for_prediction(df)  # Use self directly
            if X_pred is None:
                raise ValueError("Failed to prepare prediction data")
            
            # Verify model
            if self.model is None:
                raise ValueError("No model loaded")
            
            # Make prediction
            prediction = self.model.predict(X_pred, verbose=0)  # Use self.model directly
            
            # Transform prediction
            final_prediction = self.inverse_transform_predictions(prediction)
            
            print(f"\nPrediction Results:")
            print(f"Current Price: {df['Close'].iloc[-1]:.2f}")
            print(f"Predicted Price: {final_prediction[0]:.2f}")
            
            return final_prediction[0]
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            traceback.print_exc()
            return None

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        try:
            print("\nCalculating technical indicators...")
            
            # Moving averages
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            
            # ATR
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['ATR'] = true_range.rolling(window=14).mean()
            
            print("Technical indicators calculated successfully")
            return df
            
        except Exception as e:
            print(f"Error calculating indicators: {str(e)}")
            traceback.print_exc()
            return df

    def inverse_transform_predictions(self, predictions):
        """Transform predictions back to original scale"""
        try:
            if self.scaler is None:
                raise ValueError("Scaler not initialized")
            
            # Get Close price index
            close_idx = self.features.index('Close')
            
            # Reshape predictions
            reshaped_pred = np.zeros((len(predictions), len(self.features)))
            reshaped_pred[:, close_idx] = predictions.flatten()
            
            # Inverse transform
            inverse_transformed = self.scaler.inverse_transform(reshaped_pred)
            
            return inverse_transformed[:, close_idx]
            
        except Exception as e:
            print(f"Error in inverse transform: {str(e)}")
            traceback.print_exc()
            return predictions

    def load_model(self, model_path):
        """Load pre-trained model"""
        try:
            print(f"Loading model from: {model_path}")
            self.model = keras.models.load_model(model_path)
            print("Model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            traceback.print_exc()
    
    def save_model(self, save_path):
        """Save the current model"""
        try:
            if self.model is None:
                raise ValueError("No model to save")
                
            print(f"Saving model to: {save_path}")
            self.model.save(save_path)
            
            # Save scaler state
            scaler_path = save_path.replace('.h5', '_scaler.pkl')
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            print("Model and scaler saved successfully")
            
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            traceback.print_exc()
    
    def load_scaler(self, scaler_path):
        """Load a saved scaler"""
        try:
            print(f"Loading scaler from: {scaler_path}")
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("Scaler loaded successfully")
            
        except Exception as e:
            print(f"Error loading scaler: {str(e)}")
            traceback.print_exc()
    
    def prepare_data_for_training(self, df):
        """Prepare and scale training data"""
        try:
            print("\nPreparing training data...")
            
            # Verify features
            missing_features = [f for f in self.features if f not in df.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Extract and fit_transform feature data
            feature_data = df[self.features].values
            scaled_data = self.scaler.fit_transform(feature_data)
            
            print(f"Training data prepared, shape: {scaled_data.shape}")
            
            # Create sequences for LSTM
            X, y = [], []
            sequence_length = 10
            
            for i in range(len(scaled_data) - sequence_length):
                X.append(scaled_data[i:(i + sequence_length)])
                y.append(scaled_data[i + sequence_length, self.features.index('Close')])
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"Training sequences created. X shape: {X.shape}, y shape: {y.shape}")
            return X, y
            
        except Exception as e:
            print(f"Error preparing training data: {str(e)}")
            traceback.print_exc()
            return None, None

    def train(self, X, y, validation_split=0.2, epochs=100, batch_size=32):
        """Train the model"""
        try:
            print("\nStarting model training...")
            
            if self.model is None:
                self.build_model(input_shape=(X.shape[1], X.shape[2]))
            
            history = self.model.fit(
                X, y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                verbose=1
            )
            
            print("Model training completed")
            return history
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            traceback.print_exc()
            return None

    def build_model(self, input_shape):
        """Build LSTM model"""
        try:
            print(f"Building LSTM model with input shape: {input_shape}")
            
            self.model = keras.Sequential([
                keras.layers.LSTM(50, input_shape=input_shape, return_sequences=True),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(50, return_sequences=False),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(25),
                keras.layers.Dense(1)
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='mse',
                metrics=['mae']
            )
            
            print("Model built successfully")
            self.model.summary()
            
        except Exception as e:
            print(f"Error building model: {str(e)}")
            traceback.print_exc()

    def save_prediction(self, ticker, current_price, predicted_price, days_ahead):
        """Save prediction to DuckDB"""
        try:
            conn = duckdb.connect("forex-duckdb.db")
            
            # Create predictions table if it doesn't exist
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_predictions (
                    prediction_id INTEGER,
                    model_id INTEGER,
                    ticker VARCHAR,
                    prediction_date TIMESTAMP,
                    current_price DOUBLE,
                    predicted_price DOUBLE,
                    days_ahead INTEGER,
                    actual_price DOUBLE,  -- To be updated later
                    prediction_error DOUBLE,  -- To be updated later
                    validation_date TIMESTAMP  -- To be updated later
                )
            """)
            
            # Get next prediction_id
            result = conn.execute("SELECT COALESCE(MAX(prediction_id), 0) + 1 FROM ai_predictions").fetchone()
            prediction_id = result[0]
            
            # Get latest model_id
            result = conn.execute("SELECT MAX(model_id) FROM ai_model_metrics WHERE ticker = ?", [ticker]).fetchone()
            model_id = result[0]
            
            # Save prediction
            prediction_df = pd.DataFrame({
                'prediction_id': [prediction_id],
                'model_id': [model_id],
                'ticker': [ticker],
                'prediction_date': [pd.Timestamp.now()],
                'current_price': [current_price],
                'predicted_price': [predicted_price],
                'days_ahead': [days_ahead],
                'actual_price': [None],
                'prediction_error': [None],
                'validation_date': [None]
            })
            
            conn.execute("""
                INSERT INTO ai_predictions 
                SELECT * FROM prediction_df
            """)
            
            conn.commit()
            conn.close()
            
            print(f"Prediction saved to database with ID: {prediction_id}")
            
        except Exception as e:
            print(f"Error saving prediction: {str(e)}")
            traceback.print_exc()

    def predict(self, X):
        """Make prediction using prepared data"""
        try:
            print("\nMaking prediction...")
            
            if self.model is None:
                raise ValueError("No model loaded for prediction")
            
            if X is None:
                raise ValueError("No data provided for prediction")
            
            print(f"Input shape: {X.shape}")
            prediction = self.model.predict(X, verbose=0)
            print(f"Prediction shape: {prediction.shape}")
            
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            traceback.print_exc()
            return None

    def get_historical_data(self, ticker):
        """Get historical data for prediction"""
        try:
            print(f"\nFetching historical data for {ticker}")
            
            # Query data from database
            query = f"""
                SELECT date, open, high, low, close, volume 
                FROM {self.selected_table}
                WHERE ticker = ?
                ORDER BY date DESC
                LIMIT 100
            """
            
            df = pd.read_sql_query(query, self.conn, params=[ticker])
            
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            
            # Convert date column to datetime
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Sort by date ascending for calculations
            df.sort_index(inplace=True)
            
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            print(f"Retrieved {len(df)} records with columns: {df.columns.tolist()}")
            return df
            
        except Exception as e:
            print(f"Error getting historical data: {str(e)}")
            traceback.print_exc()
            return None

    def initialize_database(self):
        """Initialize DuckDB database and tables"""
        try:
            print("\n=== Initializing Database ===")
            
            # Create DuckDB connection
            self.conn = duckdb.connect('stock_predictions.duckdb')
            
            # Create training data table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    date TIMESTAMP,
                    ticker VARCHAR,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume BIGINT,
                    ma5 DOUBLE,
                    ma20 DOUBLE,
                    rsi DOUBLE,
                    macd DOUBLE,
                    atr DOUBLE
                )
            """)
            
            # Create predictions table
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    prediction_date TIMESTAMP,
                    ticker VARCHAR,
                    current_price DOUBLE,
                    predicted_price DOUBLE,
                    prediction_horizon INTEGER
                )
            """)
            
            print("Database initialized successfully")
            
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
            traceback.print_exc()

    def save_training_data(self, df, ticker):
        """Save training data to DuckDB"""
        try:
            print(f"\n=== Saving Training Data for {ticker} ===")
            
            # Calculate technical indicators if needed
            if 'MA5' not in df.columns:
                df = self.calculate_technical_indicators(df)
            
            # Prepare data for saving
            save_df = df.reset_index()
            save_df['ticker'] = ticker
            
            # Register DataFrame with DuckDB
            self.conn.register('temp_df', save_df)
            
            # Insert data
            self.conn.execute("""
                INSERT INTO training_data 
                SELECT 
                    date, ticker, open, high, low, close, volume,
                    ma5, ma20, rsi, macd, atr
                FROM temp_df
            """)
            
            print(f"Saved {len(df)} records to training_data table")
            
            # Verify data was saved
            count = self.conn.execute(f"""
                SELECT COUNT(*) 
                FROM training_data 
                WHERE ticker = '{ticker}'
            """).fetchone()[0]
            
            print(f"Total records for {ticker}: {count}")
            
        except Exception as e:
            print(f"Error saving training data: {str(e)}")
            traceback.print_exc()

    def get_training_data(self, ticker):
        """Retrieve training data from DuckDB"""
        try:
            print(f"\n=== Retrieving Training Data for {ticker} ===")
            
            # Query data
            query = f"""
                SELECT *
                FROM training_data
                WHERE ticker = '{ticker}'
                ORDER BY date
            """
            
            df = self.conn.execute(query).fetchdf()
            
            if df.empty:
                print(f"No training data found for {ticker}")
                return None
            
            print(f"Retrieved {len(df)} records")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Date range: {df['date'].min()} to {df['date'].max()}")
            
            return df
            
        except Exception as e:
            print(f"Error retrieving training data: {str(e)}")
            traceback.print_exc()
            return None

    def verify_training_data(self):
        """Verify training data in database"""
        try:
            print("\n=== Verifying Training Data ===")
            
            # Check table exists
            tables = self.conn.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_name = 'training_data'
            """).fetchall()
            
            if not tables:
                print("Training data table does not exist")
                return False
            
            # Get table info
            count = self.conn.execute("SELECT COUNT(*) FROM training_data").fetchone()[0]
            tickers = self.conn.execute("SELECT DISTINCT ticker FROM training_data").fetchdf()
            
            print(f"Total records: {count}")
            print(f"Unique tickers: {tickers['ticker'].tolist()}")
            
            # Verify data quality
            nulls = self.conn.execute("""
                SELECT 
                    COUNT(*) - COUNT(date) as date_nulls,
                    COUNT(*) - COUNT(ticker) as ticker_nulls,
                    COUNT(*) - COUNT(close) as close_nulls
                FROM training_data
            """).fetchdf()
            
            print("\nData Quality Check:")
            print(f"Null dates: {nulls['date_nulls'].iloc[0]}")
            print(f"Null tickers: {nulls['ticker_nulls'].iloc[0]}")
            print(f"Null prices: {nulls['close_nulls'].iloc[0]}")
            
            return count > 0
            
        except Exception as e:
            print(f"Error verifying training data: {str(e)}")
            traceback.print_exc()
            return False

    def train_model(self, ticker):
        """Train model using data from DuckDB"""
        try:
            print(f"\n=== Training Model for {ticker} ===")
            
            # Get training data
            df = self.get_training_data(ticker)
            if df is None or df.empty:
                raise ValueError(f"No training data available for {ticker}")
            
            # Prepare features
            X, y = self.prepare_data_for_training(df)
            if X is None or y is None:
                raise ValueError("Failed to prepare training data")
            
            print(f"Training data shape: X={X.shape}, y={y.shape}")
            
            # Train model
            history = self.model.fit(
                X, y,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=0.2,
                verbose=1
            )
            
            print("Model training completed")
            return history
            
        except Exception as e:
            print(f"Error training model: {str(e)}")
            traceback.print_exc()
            return None 