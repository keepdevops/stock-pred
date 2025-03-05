import os
# Set environment variable to silence Tk deprecation warning
os.environ['TK_SILENCE_DEPRECATION'] = '1'  # Silence Tk deprecation warning

# Force CPU usage for TensorFlow to avoid Metal GPU issues
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Hide all GPUs
os.environ['TF_DISABLE_GRAPPLER'] = '1'    # Disable the Grappler optimizer
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'false'  # Disable memory growth

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import Huber
from sqlalchemy import create_engine, text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.dates as mdates
import matplotlib
import tkinter as tk
from tkinter import ttk, messagebox
import traceback
from datetime import datetime
import glob
from sklearn.preprocessing import MinMaxScaler
import pickle
import json

# Set Matplotlib backend
plt.switch_backend('TkAgg')

# Global variables for trained model and scaler
trained_model = None
trained_scaler = None

# Also add this after importing TensorFlow:
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Hide all GPUs from TensorFlow
tf.keras.backend.set_floatx('float32')    # Use float32 for better CPU performance

def find_databases():
    """Find all database files in the current directory"""
    return glob.glob('*.db') + glob.glob('*.duckdb')

def create_connection(db_name):
    """Create a database connection using SQLAlchemy"""
    try:
        engine = create_engine(f"duckdb:///{db_name}")
        return engine
    except Exception as e:
        print(f"Error connecting to database {db_name}: {e}")
        return None

class DataAdapter:
    def __init__(self, sequence_length=10, features=None):
        self.sequence_length = sequence_length
        self.features = features or ['open', 'high', 'low', 'close', 'volume', 'ma20', 'ma50', 'rsi', 'macd']
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def prepare_training_data(self, data):
        """Prepare data for training by creating sequences"""
        try:
            print("\nPreparing training data...")
            df = data.copy()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date').sort_index()

            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            df = df[self.features].ffill().bfill().fillna(0)
            scaled_data = self.scaler.fit_transform(df)

            X, y = [], []
            for i in range(len(scaled_data) - self.sequence_length):
                X.append(scaled_data[i:i + self.sequence_length])
                y.append(scaled_data[i + self.sequence_length, 3])  # 'close' at index 3
            X, y = np.array(X), np.array(y)

            train_size = int(len(X) * 0.8)
            X_train, X_val = X[:train_size], X[train_size:]
            y_train, y_val = y[:train_size], y[train_size:]

            print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
            return X_train, X_val, y_train, y_val
        except Exception as e:
            print(f"Error preparing training data: {e}")
            traceback.print_exc()
            return None, None, None, None

    def prepare_prediction_data(self, data):
        """Prepare data for prediction"""
        try:
            print("\nPreparing prediction data...")
            df = data.copy()
            df['date'] = pd.to_datetime(df['date'])
            
            # Sort by date
            if 'date' in df.columns:
                df = df.sort_values('date')
            
            # Calculate technical indicators if needed
            df = self.calculate_technical_indicators(df)
            
            # Make sure we have all required features
            for feature in self.features:
                if feature not in df.columns and feature != 'date':
                    print(f"Warning: Feature '{feature}' not found in data. Adding zeros.")
                    df[feature] = 0
                    
            # Use exactly the line provided for NaN handling
            df = df[self.features].ffill().bfill().fillna(0)
            
            # Scale the data
            scaled_data = self.scaler.transform(df)

            # Create sequences
            X = []
            for i in range(len(scaled_data) - self.sequence_length + 1):
                X.append(scaled_data[i:i + self.sequence_length])
            return np.array(X) if X else None
            
        except Exception as e:
            print(f"Error preparing prediction data: {e}")
            traceback.print_exc()
            return None

    def inverse_transform_predictions(self, predictions):
        """Convert scaled predictions back to original scale"""
        try:
            dummy = np.zeros((len(predictions), len(self.features)))
            dummy[:, 3] = predictions.flatten()  # 'close' at index 3
            return self.scaler.inverse_transform(dummy)[:, 3]
        except Exception as e:
            print(f"Error inverse transforming predictions: {e}")
            return None

    def calculate_technical_indicators(self, df):
        """Calculate technical indicators"""
        try:
            if 'close' in df.columns:
                df['ma20'] = df['close'].rolling(window=20).mean()
                df['ma50'] = df['close'].rolling(window=50).mean()
                delta = df['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                rs = gain / loss.replace(0, 0.001)
                df['rsi'] = 100 - (100 / (1 + rs))
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                df['macd'] = ema12 - ema26
            return df.ffill().bfill().fillna(0)
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return df

class StockAIAgent:
    def __init__(self):
        self.model = None
        self.sequence_length = 10
        self.data_adapter = DataAdapter(sequence_length=self.sequence_length)
        self.learning_rate = 0.001

    def build_model(self, input_shape):
        """Build and compile the LSTM model"""
        try:
            print(f"Building model with input shape: {input_shape}")
            model = Sequential([
                Input(shape=input_shape),
                LSTM(50, return_sequences=True),
                BatchNormalization(),
                Dropout(0.2),
                LSTM(50, return_sequences=True),
                BatchNormalization(),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                BatchNormalization(),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss=Huber(), metrics=['mae'])
            self.model = model
            print("Model built successfully")
            model.summary()
            return model
        except Exception as e:
            print(f"Error building model: {e}")
            traceback.print_exc()
            return None

    def train(self, df, epochs=50, batch_size=32):
        """Train the LSTM model"""
        try:
            print("\nStarting model training...")
            data = self.data_adapter.prepare_training_data(df)
            if data is None or any(x is None for x in data):
                raise ValueError("Data preparation failed")

            X_train, X_val, y_train, y_val = data
            print(f"Training data shapes: X_train={X_train.shape}, y_train={y_train.shape}")
            print(f"Validation data shapes: X_val={X_val.shape}, y_val={y_val.shape}")

            if self.model is None:
                input_shape = (X_train.shape[1], X_train.shape[2])
                self.build_model(input_shape)

            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
                verbose=1
            )
            print("Model training completed")
            return history
        except Exception as e:
            print(f"Error training model: {e}")
            traceback.print_exc()
            return None

    def predict(self, df, future_only=False):
        """Make predictions using the trained model"""
        try:
            print("\nStarting prediction...")
            if self.model is None:
                raise ValueError("Model not trained")

            X = self.data_adapter.prepare_prediction_data(df)
            if X is None:
                raise ValueError("Prediction data preparation failed")

            if future_only:
                X = X[-1:]  # Use only the last sequence

            predictions = self.model.predict(X)
            inverse_predictions = self.data_adapter.inverse_transform_predictions(predictions)
            return inverse_predictions[0] if future_only else inverse_predictions
        except Exception as e:
            print(f"Error making predictions: {e}")
            traceback.print_exc()
            return None

    def predict_future(self, df, days=30):
        """Predict future prices for multiple days ahead"""
        try:
            predictions = []
            temp_df = df.copy()

            for _ in range(days):
                next_price = self.predict(temp_df, future_only=True)
                if next_price is None:
                    break
                predictions.append(next_price)

                last_date = pd.to_datetime(temp_df['date'].iloc[-1])
                next_date = last_date + pd.Timedelta(days=1)
                new_row = pd.DataFrame({
                    'date': [next_date],
                    'open': [next_price],
                    'high': [next_price * 1.005],  # Slight increase
                    'low': [next_price * 0.995],   # Slight decrease
                    'close': [next_price],
                    'volume': [temp_df['volume'].mean()],
                    'ticker': [temp_df['ticker'].iloc[-1]]
                })
                temp_df = pd.concat([temp_df, new_row], ignore_index=True)

            return predictions
        except Exception as e:
            print(f"Error in future prediction: {e}")
            traceback.print_exc()
            return []

def create_lstm_model(input_shape, neurons=50, num_layers=2, dropout_rate=0.2):
    """Create an LSTM model with the specified architecture"""
    model = tf.keras.Sequential()
    
    # First layer with explicit Input shape
    model.add(tf.keras.layers.Input(shape=input_shape))  # Use 'shape' here
    model.add(tf.keras.layers.LSTM(neurons, return_sequences=(num_layers > 1)))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Additional LSTM layers if requested
    for i in range(1, num_layers):
        return_sequences = i < num_layers - 1  # Only last layer doesn't return sequences
        model.add(tf.keras.layers.LSTM(neurons, return_sequences=return_sequences))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(tf.keras.layers.Dense(1))
    return model

def create_gru_model(input_shape, neurons=50, num_layers=2, dropout_rate=0.2):
    """Create a GRU model with the specified architecture"""
    model = tf.keras.Sequential()
    
    # First layer with explicit Input shape
    model.add(tf.keras.layers.Input(shape=input_shape))  # Use 'shape' here
    model.add(tf.keras.layers.GRU(neurons, return_sequences=(num_layers > 1)))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Additional GRU layers if requested
    for i in range(1, num_layers):
        return_sequences = i < num_layers - 1  # Only last layer doesn't return sequences
        model.add(tf.keras.layers.GRU(neurons, return_sequences=return_sequences))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(tf.keras.layers.Dense(1))
    return model

def create_bilstm_model(input_shape, neurons=50, num_layers=2, dropout_rate=0.2):
    """Create a Bidirectional LSTM model with the specified architecture"""
    model = tf.keras.Sequential()
    
    # First layer with explicit Input shape
    model.add(tf.keras.layers.Input(shape=input_shape))  # Use 'shape' here
    model.add(tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(neurons, return_sequences=(num_layers > 1))
    ))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Additional BiLSTM layers if requested
    for i in range(1, num_layers):
        return_sequences = i < num_layers - 1  # Only last layer doesn't return sequences
        model.add(tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(neurons, return_sequences=return_sequences)
        ))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(tf.keras.layers.Dense(1))
    return model

def create_cnn_lstm_model(input_shape, filters=64, kernel_size=3, lstm_units=50, dropout_rate=0.2):
    """Create a CNN-LSTM model with the specified architecture"""
    model = tf.keras.Sequential()
    
    # CNN layers with explicit Input shape
    model.add(tf.keras.layers.Input(shape=input_shape))  # Use 'shape' here
    model.add(tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size, 
                                     activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # LSTM layer
    model.add(tf.keras.layers.LSTM(lstm_units))
    model.add(tf.keras.layers.Dropout(dropout_rate))
    
    # Output layer
    model.add(tf.keras.layers.Dense(1))
    return model

def configure_black_text_widgets():
    """Configure all input widgets to have black text"""
    style = ttk.Style()
    
    # Configure Entry widgets to have black text
    style.map("TEntry", 
        foreground=[("active", "black"), ("disabled", "gray"), ("!disabled", "black")],
        fieldbackground=[("!disabled", "white")]
    )
    
    # Configure Combobox widgets to have black text
    style.map("TCombobox",
        foreground=[("active", "black"), ("disabled", "gray"), ("!disabled", "black")],
        fieldbackground=[("!disabled", "white")]
    )
    
    # Configure Listbox text color
    root.option_add("*TListbox*foreground", "black")
    
    # Configure dropdown lists in comboboxes to have black text
    root.option_add('*TCombobox*Listbox.foreground', 'black')
    
    # Make the selected text in comboboxes black
    style.map('TCombobox', 
              foreground=[('readonly', 'black'), ('active', 'black'), ('disabled', 'gray')],
              fieldbackground=[('readonly', 'white'), ('active', 'white'), ('disabled', 'gray')])
    
    # Configure Spinbox widgets to have black text
    style.configure("BlackText.TSpinbox", foreground="black")
    style.map("BlackText.TSpinbox",
              foreground=[('readonly', 'black'), ('disabled', 'gray'), ('active', 'black')],
              fieldbackground=[('readonly', 'white'), ('disabled', '#D3D3D3'), ('active', 'white')])
    
    return style

def initialize_control_panel(main_frame, databases):
    """Initialize the control panel with database, table, and ticker controls"""
    try:
        print("Creating control panel...")
        control_panel = ttk.Frame(main_frame)
        control_panel.pack(side="left", fill="y", padx=10, pady=10)

        # Create a scrollable container
        control_container = ttk.Frame(control_panel)
        control_container.pack(fill="both", expand=True)
        
        canvas = tk.Canvas(control_container)
        scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=canvas.yview)
        
        scrollable_control_frame = ttk.Frame(canvas)
        scrollable_control_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_control_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Initialize variables
        global status_var, db_var, table_var, days_var, epochs_var, batch_size_var, learning_rate_var, sequence_length_var, ticker_listbox, output_text
        global model_type_var, num_layers_var, neurons_var, dropout_var
        status_var = tk.StringVar(value="Ready")
        db_var = tk.StringVar()
        table_var = tk.StringVar()
        days_var = tk.IntVar(value=30)
        epochs_var = tk.IntVar(value=50)
        batch_size_var = tk.IntVar(value=32)
        learning_rate_var = tk.DoubleVar(value=0.001)
        sequence_length_var = tk.IntVar(value=10)
        
        # Model architecture variables
        model_type_var = tk.StringVar(value="LSTM")
        num_layers_var = tk.IntVar(value=2)
        neurons_var = tk.IntVar(value=50)
        dropout_var = tk.DoubleVar(value=0.2)

        # Database selection
        db_frame = ttk.LabelFrame(scrollable_control_frame, text="Database Selection")
        db_frame.pack(fill="x", padx=5, pady=5)
        db_combo = ttk.Combobox(db_frame, textvariable=db_var, state="readonly", width=28)
        db_combo["values"] = databases
        if databases:
            db_var.set(databases[0])
        db_combo.pack(fill="x", padx=5, pady=5)

        # Table selection
        table_frame = ttk.LabelFrame(scrollable_control_frame, text="Table Selection")
        table_frame.pack(fill="x", padx=5, pady=5)
        table_combo = ttk.Combobox(table_frame, textvariable=table_var, state="readonly", width=28)
        table_combo.pack(fill="x", padx=5, pady=5)

        # Ticker selection
        ticker_frame = ttk.LabelFrame(scrollable_control_frame, text="Ticker Selection")
        ticker_frame.pack(fill="x", padx=5, pady=5)
        
        # Create a container for the listbox and scrollbar
        ticker_list_container = ttk.Frame(ticker_frame)
        ticker_list_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create the listbox with reduced height to make room for buttons
        ticker_listbox = tk.Listbox(ticker_list_container, selectmode="multiple", height=5, bg="#FFFFFF", fg="black", width=30)
        ticker_scrollbar = ttk.Scrollbar(ticker_list_container, orient="vertical", command=ticker_listbox.yview)
        ticker_listbox.configure(yscrollcommand=ticker_scrollbar.set)
        ticker_scrollbar.pack(side="right", fill="y")
        ticker_listbox.pack(side="left", fill="both", expand=True)
        
        # Add selection control buttons in their own frame with more space
        ticker_buttons_frame = ttk.Frame(ticker_frame)
        ticker_buttons_frame.pack(fill="x", padx=5, pady=5)  # Increased padding
        
        def select_all_tickers():
            ticker_listbox.select_set(0, tk.END)
        
        def clear_all_tickers():
            ticker_listbox.selection_clear(0, tk.END)
        
        # Create a more spacious button layout
        select_all_button = ttk.Button(ticker_buttons_frame, text="Select All", command=select_all_tickers, width=12)
        select_all_button.pack(side="left", padx=5, pady=3, expand=True, fill="x")
        
        clear_all_button = ttk.Button(ticker_buttons_frame, text="Clear All", command=clear_all_tickers, width=12)
        clear_all_button.pack(side="right", padx=5, pady=3, expand=True, fill="x")

        # Duration controls
        duration_frame = ttk.LabelFrame(scrollable_control_frame, text="Prediction Duration")
        duration_frame.pack(fill="x", padx=5, pady=5)
        days_combo = ttk.Combobox(duration_frame, textvariable=days_var, values=[1, 7, 14, 30, 60, 90], state="readonly", width=28)
        days_combo.pack(fill="x", padx=5, pady=5)

        # AI model architecture controls
        model_frame = ttk.LabelFrame(scrollable_control_frame, text="Model Architecture")
        model_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(model_frame, text="Model Type:").pack(anchor="w", padx=5)
        model_combo = ttk.Combobox(model_frame, textvariable=model_type_var, 
                                  values=["LSTM", "GRU", "BiLSTM", "CNN-LSTM"],
                                  state="readonly", width=28)
        model_combo.config(foreground="black")  # This sets the text color
        model_combo.option_add('*TCombobox*Listbox.foreground', 'black')  # This sets dropdown list text color
        model_combo.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(model_frame, text="Number of Layers:").pack(anchor="w", padx=5)
        layers_spin = ttk.Spinbox(model_frame, from_=1, to=5, textvariable=num_layers_var, width=28, style="BlackText.TSpinbox")
        layers_spin.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(model_frame, text="Neurons per Layer:").pack(anchor="w", padx=5)
        neurons_spin = ttk.Spinbox(model_frame, from_=10, to=200, textvariable=neurons_var, width=28, style="BlackText.TSpinbox")
        neurons_spin.pack(fill="x", padx=5, pady=2)
        
        ttk.Label(model_frame, text="Dropout Rate:").pack(anchor="w", padx=5)
        dropout_spin = ttk.Spinbox(model_frame, from_=0.0, to=0.5, increment=0.1, textvariable=dropout_var, width=28, style="BlackText.TSpinbox")
        dropout_spin.pack(fill="x", padx=5, pady=2)

        # AI training controls
        ai_frame = ttk.LabelFrame(scrollable_control_frame, text="Training Parameters")
        ai_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(ai_frame, text="Epochs:").pack(anchor="w", padx=5)
        epochs_entry = ttk.Entry(ai_frame, textvariable=epochs_var, width=28, style="BlackText.TEntry")
        epochs_entry.pack(fill="x", padx=5, pady=2)
        ttk.Label(ai_frame, text="Batch Size:").pack(anchor="w", padx=5)
        batch_size_entry = ttk.Entry(ai_frame, textvariable=batch_size_var, width=28, style="BlackText.TEntry")
        batch_size_entry.pack(fill="x", padx=5, pady=2)
        ttk.Label(ai_frame, text="Learning Rate:").pack(anchor="w", padx=5)
        learning_rate_entry = ttk.Entry(ai_frame, textvariable=learning_rate_var, width=28, style="BlackText.TEntry")
        learning_rate_entry.pack(fill="x", padx=5, pady=2)
        ttk.Label(ai_frame, text="Sequence Length:").pack(anchor="w", padx=5)
        sequence_length_entry = ttk.Entry(ai_frame, textvariable=sequence_length_var, width=28, style="BlackText.TEntry")
        sequence_length_entry.pack(fill="x", padx=5, pady=2)

        # Buttons
        buttons_frame = ttk.Frame(scrollable_control_frame)
        buttons_frame.pack(fill="x", padx=5, pady=5)
        train_button = ttk.Button(buttons_frame, text="Train Model", command=train_model_handler)
        train_button.pack(side="left", fill="x", expand=True, padx=2)
        predict_button = ttk.Button(buttons_frame, text="Make Prediction", command=predict_handler)
        predict_button.pack(side="right", fill="x", expand=True, padx=2)

        # Model save/load buttons
        model_io_frame = ttk.Frame(scrollable_control_frame)
        model_io_frame.pack(fill="x", padx=5, pady=5)
        save_button = ttk.Button(model_io_frame, text="Save Model", command=save_model_handler)
        save_button.pack(side="left", fill="x", expand=True, padx=2)
        load_button = ttk.Button(model_io_frame, text="Load Model", command=load_model_handler)
        load_button.pack(side="right", fill="x", expand=True, padx=2)

        # Output area
        output_frame = ttk.LabelFrame(scrollable_control_frame, text="Results")
        output_frame.pack(fill="both", expand=True, padx=5, pady=5)
        output_text = tk.Text(output_frame, height=10, bg="#FFFFFF", fg="black", width=30)
        output_scrollbar = ttk.Scrollbar(output_frame, command=output_text.yview)
        output_text.configure(yscrollcommand=output_scrollbar.set)
        output_scrollbar.pack(side="right", fill="y")
        output_text.pack(side="left", fill="both", expand=True)

        # Status bar
        status_bar = ttk.Label(control_panel, textvariable=status_var, relief="sunken")
        status_bar.pack(side="bottom", fill="x")

        # Event handlers
        def on_database_selected(event=None):
            db_name = db_var.get()
            if db_name:
                tables = get_tables(db_name)
                table_combo["values"] = tables
                if tables:
                    table_var.set(tables[0])
                    on_table_selected()
                else:
                    table_var.set("")
                    ticker_listbox.delete(0, tk.END)
        
        def on_table_selected(event=None):
            db_name = db_var.get()
            table_name = table_var.get()
            if db_name and table_name:
                load_tickers(db_name, table_name, ticker_listbox)

        db_combo.bind("<<ComboboxSelected>>", on_database_selected)
        table_combo.bind("<<ComboboxSelected>>", on_table_selected)

        on_database_selected()
        print("AI controls setup complete")
        return {'control_panel': control_panel, 'output_text': output_text, 'status_bar': status_bar}

    except Exception as e:
        print(f"Error initializing control panel: {e}")
        traceback.print_exc()
        raise

def initialize_gui(databases):
    """Initialize the GUI components"""
    try:
        print("\nStarting initialize_gui...")
        root = tk.Tk()
        root.title("Stock Market Analyzer")
        root.geometry('1200x800')

        style = ttk.Style()
        style.theme_use('clam')
        style.configure(".", background="#2E2E2E", foreground="#FFFFFF")
        style.configure("TFrame", background="#2E2E2E")
        style.configure("TLabel", background="#2E2E2E", foreground="#FFFFFF")
        style.configure("TButton", background="#3E3E3E", foreground="#FFFFFF")
        
        # Configure Combobox with black text for the selected value
        style.configure("TCombobox", fieldbackground="white", foreground="black")
        style.map('TCombobox', 
                 fieldbackground=[('readonly', 'white')],
                 foreground=[('readonly', 'black')],
                 selectbackground=[('readonly', '#0078D7')], 
                 selectforeground=[('readonly', 'white')])
        
        # Create a custom style for Entry widgets with black text on white background
        style.configure("BlackText.TEntry", foreground="black")
        
        # Configure listbox and combobox dropdown selection colors for better contrast
        root.option_add('*TCombobox*Listbox.background', '#FFFFFF')
        root.option_add('*TCombobox*Listbox.foreground', 'black')
        root.option_add('*TCombobox*Listbox.selectBackground', '#0078D7')
        root.option_add('*TCombobox*Listbox.selectForeground', 'white')
        root.option_add('*Listbox.selectBackground', '#0078D7')  # Windows-style selection blue
        root.option_add('*Listbox.selectForeground', 'white')    # White text on selection

        main_frame = ttk.Frame(root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        control_frame = initialize_control_panel(main_frame, databases)

        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        global fig, canvas
        fig = Figure(figsize=(8, 6), dpi=100)
        fig.patch.set_facecolor("#2E2E2E")
        ax = fig.add_subplot(111)
        ax.set_facecolor("#3E3E3E")
        ax.tick_params(colors="white")
        ax.spines["bottom"].set_color("white")
        ax.spines["top"].set_color("white")
        ax.spines["left"].set_color("white")
        ax.spines["right"].set_color("white")

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        print("Successfully completed initialize_gui")
        return root
    except Exception as e:
        print(f"Error in initialize_gui: {e}")
        traceback.print_exc()
        raise

def save_model(model, scaler, ticker, model_path="models"):
    """Save the trained model and scaler"""
    try:
        os.makedirs(model_path, exist_ok=True)
        
        # Use .keras extension as requested
        model.save(os.path.join(model_path, f"{ticker}_model.keras"))
        
        with open(os.path.join(model_path, f"{ticker}_scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Model and scaler saved for {ticker}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def save_model_handler():
    """Handle saving model with metadata"""
    if not trained_model:
        messagebox.showerror("Error", "No trained model available to save")
        return
    
    selected_indices = ticker_listbox.curselection()
    if not selected_indices:
        messagebox.showerror("Error", "Please select at least one ticker")
        return
        
    ticker = ticker_listbox.get(selected_indices[0])
    model_type = model_type_var.get()
    
    try:
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{ticker}_{model_type.lower()}_model.keras")
        scaler_path = os.path.join(model_dir, f"{ticker}_{model_type.lower()}_scaler.pkl")
        
        # Save model and scaler
        trained_model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(trained_scaler, f)
            
        # Save metadata
        metadata = {
            "ticker": ticker,
            "model_type": model_type,
            "sequence_length": sequence_length_var.get(),
            "epochs": epochs_var.get(),
            "batch_size": batch_size_var.get(),
            "num_layers": num_layers_var.get(),
            "neurons": neurons_var.get(),
            "dropout": dropout_var.get(),
            "saved_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(model_dir, f"{ticker}_{model_type.lower()}_metadata.json"), 'w') as f:
            json.dump(metadata, f)
            
        status_var.set(f"Model saved for {ticker}")
        output_text.insert("end", f"Model saved to {model_path}\n")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to save model: {str(e)}")
        traceback.print_exc()

def load_model_handler():
    """Handle loading a previously saved model"""
    global trained_model, trained_scaler
    
    selected_indices = ticker_listbox.curselection()
    if not selected_indices:
        messagebox.showerror("Error", "Please select at least one ticker")
        return
        
    ticker = ticker_listbox.get(selected_indices[0])
    model_type = model_type_var.get()
    
    try:
        model_dir = "models"
        model_path = os.path.join(model_dir, f"{ticker}_{model_type.lower()}_model.keras")
        scaler_path = os.path.join(model_dir, f"{ticker}_{model_type.lower()}_scaler.pkl")
        metadata_path = os.path.join(model_dir, f"{ticker}_{model_type.lower()}_metadata.json")
        
        # Check if all required files exist
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"No model found for {ticker} ({model_type}) at {model_path}")
            return
        if not os.path.exists(scaler_path):
            messagebox.showerror("Error", f"No scaler found for {ticker} ({model_type}) at {scaler_path}")
            return
        if not os.path.exists(metadata_path):
            messagebox.showwarning("Warning", f"No metadata found for {ticker} ({model_type}) at {metadata_path}. Loading model without metadata.")

        # Load the model
        trained_model = tf.keras.models.load_model(model_path)
        
        # Load the scaler
        with open(scaler_path, 'rb') as f:
            trained_scaler = pickle.load(f)
        
        # Load and display metadata if available
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            output_text.insert("end", f"Loaded model metadata:\n")
            for key, value in metadata.items():
                output_text.insert("end", f"- {key}: {value}\n")
        
        status_var.set(f"Model loaded for {ticker} ({model_type})")
        output_text.insert("end", f"Model loaded from {model_path}\n")
        output_text.insert("end", f"Scaler loaded from {scaler_path}\n")
        
        # Optionally update UI variables based on metadata
        if metadata:
            sequence_length_var.set(metadata.get("sequence_length", sequence_length_var.get()))
            epochs_var.set(metadata.get("epochs", epochs_var.get()))
            batch_size_var.set(metadata.get("batch_size", batch_size_var.get()))
            num_layers_var.set(metadata.get("num_layers", num_layers_var.get()))
            neurons_var.set(metadata.get("neurons", neurons_var.get()))
            dropout_var.set(metadata.get("dropout", dropout_var.get()))
            output_text.insert("end", "Updated training parameters from metadata\n")
        
    except Exception as e:
        status_var.set("Model loading failed")
        messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        traceback.print_exc()

def train_model_handler():
    global trained_model, trained_scaler
    status_var.set("Training model...")
    output_text.delete(1.0, "end")

    db_name = db_var.get()
    table_name = table_var.get()
    selected_indices = ticker_listbox.curselection()
    tickers = [ticker_listbox.get(i) for i in selected_indices]
    
    if not tickers:
        error_msg = "Please select at least one ticker"
        status_var.set(error_msg)
        output_text.insert("end", f"Error: {error_msg}\n")
        messagebox.showerror("Error", error_msg)
        return
    
    output_text.insert("end", f"Selected {len(tickers)} tickers for training\n")
    output_text.update()
    
    if len(tickers) > 10:
        warning_msg = f"You've selected {len(tickers)} tickers. Training with many tickers may take longer."
        output_text.insert("end", f"Warning: {warning_msg}\n")
        output_text.update()
        if not messagebox.askyesno("Warning", f"{warning_msg}\n\nDo you want to continue?"):
            status_var.set("Training cancelled")
            return

    engine = create_connection(db_name)
    if not engine:
        error_msg = f"Could not connect to database: {db_name}"
        status_var.set(error_msg)
        output_text.insert("end", f"Error: {error_msg}\n")
        return

    try:
        placeholders = ', '.join([':ticker' + str(i) for i in range(len(tickers))])
        query = text(f"SELECT * FROM {table_name} WHERE ticker IN ({placeholders})")
        params = {f'ticker{i}': ticker for i, ticker in enumerate(tickers)}
        
        output_text.insert("end", "Fetching data from database...\n")
        output_text.update()
        
        with engine.connect() as conn:
            result = conn.execute(query, params)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())

        if df.empty:
            error_msg = f"No data found for selected tickers"
            status_var.set(error_msg)
            output_text.insert("end", f"Error: {error_msg}\n")
            return

        output_text.insert("end", f"Fetched {len(df)} rows of data\n")
        output_text.insert("end", f"Date range: {df['date'].min()} to {df['date'].max()}\n")
        output_text.insert("end", f"Tickers in dataset: {', '.join(df['ticker'].unique())}\n\n")
        output_text.update()

        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            error_msg = f"Table '{table_name}' is missing required columns: {', '.join(missing_columns)}."
            status_var.set(error_msg)
            output_text.insert("end", f"Error: {error_msg}\n")
            return

        df['date'] = pd.to_datetime(df['date'])
        
        sequence_length = sequence_length_var.get()
        epochs = epochs_var.get()
        batch_size = batch_size_var.get()
        learning_rate = learning_rate_var.get()
        model_type = model_type_var.get()
        num_layers = num_layers_var.get()
        neurons = neurons_var.get()
        dropout_rate = dropout_var.get()
        
        output_text.insert("end", f"Starting model training with:\n")
        output_text.insert("end", f"- Model type: {model_type}\n")
        output_text.insert("end", f"- Layers: {num_layers}\n")
        output_text.insert("end", f"- Neurons: {neurons}\n")
        output_text.insert("end", f"- Sequence length: {sequence_length}\n")
        output_text.insert("end", f"- Epochs: {epochs}\n")
        output_text.insert("end", f"- Batch size: {batch_size}\n")
        output_text.insert("end", f"- Tickers: {', '.join(tickers)}\n\n")
        output_text.update()
        
        data_adapter = DataAdapter(sequence_length=sequence_length)
        X_train, X_val, y_train, y_val = data_adapter.prepare_training_data(df)
        
        if X_train is None or y_train is None:
            status_var.set("Training data preparation failed")
            output_text.insert("end", "Training data preparation failed\n")
            return
        
        output_text.insert("end", f"Data prepared: {X_train.shape[0]} training samples\n")
        output_text.update()
        
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        if model_type == "LSTM":
            model = create_lstm_model(input_shape, neurons, num_layers, dropout_rate)
        elif model_type == "GRU":
            model = create_gru_model(input_shape, neurons, num_layers, dropout_rate)
        elif model_type == "BiLSTM":
            model = create_bilstm_model(input_shape, neurons, num_layers, dropout_rate)
        elif model_type == "CNN-LSTM":
            model = create_cnn_lstm_model(input_shape, filters=neurons, lstm_units=neurons, dropout_rate=dropout_rate)
        else:
            model = create_lstm_model(input_shape, neurons, num_layers, dropout_rate)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        output_text.insert("end", "Training model... This may take a while.\n")
        output_text.update()
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        trained_model = model
        trained_scaler = data_adapter.scaler
        
        # Predict historical data for all tickers
        agent = StockAIAgent()
        agent.model, agent.data_adapter.scaler = trained_model, trained_scaler
        agent.sequence_length = sequence_length
        
        historical_predictions = agent.predict(df, future_only=False)
        historical_dates = df['date'].iloc[sequence_length:]
        historical_predictions = historical_predictions[:len(historical_dates)]
        
        # Calculate metrics and plot
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor("#3E3E3E")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_color("white")
        
        ticker_metrics = {}
        colors = plt.cm.tab20  # Access colormap through plt.cm
        
        for i, ticker in enumerate(tickers):
            ticker_df = df[df['ticker'] == ticker]
            ticker_dates = ticker_df['date'].iloc[sequence_length:]
            ticker_close = ticker_df['close'].iloc[sequence_length:]
            ticker_pred = historical_predictions[ticker_df.index[sequence_length:]-sequence_length]
            
            # Plot with color index
            color = colors(i / len(tickers))  # Normalize index to [0, 1] for colormap
            ax.plot(ticker_df['date'], ticker_df['close'], label=f'{ticker} Historical', color=color)
            ax.plot(ticker_dates, ticker_pred, '--', label=f'{ticker} Predicted', color=color)
            
            # Normalization percentage
            start_price = ticker_df['close'].iloc[0]
            end_price = ticker_pred[-1] if len(ticker_pred) > 0 else ticker_close.iloc[-1]
            norm_pct = ((end_price - start_price) / start_price) * 100
            
            # Return in dollars (assume $1,000 initial investment)
            initial_investment = 1000
            shares = initial_investment / start_price
            dollar_return = shares * (end_price - start_price)
            
            ticker_metrics[ticker] = {'norm_pct': norm_pct, 'dollar_return': dollar_return}
        
        # Rank tickers by normalization percentage
        ranked_tickers = sorted(ticker_metrics.items(), key=lambda x: x[1]['norm_pct'], reverse=True)
        
        ax.set_title("Historical Data and Predictions", color="white")
        ax.set_xlabel('Date', color="white")
        ax.set_ylabel('Price', color="white")
        ax.legend(facecolor="#2E2E2E", labelcolor="white", loc='best')
        fig.autofmt_xdate()
        canvas.draw()
        
        # Display metrics
        output_text.insert("end", "Training Results:\n")
        output_text.insert("end", f"Trained for {len(history.history['loss'])} epochs\n")
        output_text.insert("end", f"Final training loss: {history.history['loss'][-1]:.6f}\n")
        output_text.insert("end", f"Final validation loss: {history.history['val_loss'][-1]:.6f}\n\n")
        output_text.insert("end", "Ticker Performance Metrics:\n")
        for ticker, metrics in ranked_tickers:
            output_text.insert("end", f"{ticker}:\n")
            output_text.insert("end", f"  Normalization %: {metrics['norm_pct']:.2f}%\n")
            output_text.insert("end", f"  Return ($1,000 invested): ${metrics['dollar_return']:.2f}\n")
        output_text.insert("end", "\nBest Ticker to Invest In (by Normalization %):\n")
        output_text.insert("end", f"1. {ranked_tickers[0][0]} ({ranked_tickers[0][1]['norm_pct']:.2f}%)\n")
        
        status_var.set("Model training completed successfully")
        
    except Exception as e:
        status_var.set("Model training failed")
        output_text.insert("end", f"Error during training: {str(e)}\n")
        traceback.print_exc()
    finally:
        engine.dispose()

def load_tables(db_name, table_combo):
    """Load tables from the specified database"""
    try:
        tables = get_tables(db_name)
        table_combo["values"] = tables
        if tables:
            table_var.set(tables[0])
            # Update ticker list based on the selected table
            load_tickers(db_name, table_var.get(), ticker_listbox)
    except Exception as e:
        print(f"Error loading tables: {e}")
        messagebox.showerror("Error", f"Failed to load tables: {str(e)}")

def get_tables(db_name):
    """Get list of tables from the specified database"""
    engine = create_connection(db_name)
    if not engine:
        return []
    
    try:
        with engine.connect() as conn:
            # For SQLite
            if db_name.endswith('.db'):
                query = text("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in conn.execute(query)]
            # For DuckDB
            elif db_name.endswith('.duckdb'):
                query = text("SELECT table_name FROM information_schema.tables WHERE table_schema='main'")
                tables = [row[0] for row in conn.execute(query)]
            else:
                tables = []
        return tables
    except Exception as e:
        print(f"Error getting tables: {e}")
        return []
    finally:
        engine.dispose()

def load_tickers(db_name, table_name, ticker_listbox):
    """Load tickers from the specified table"""
    engine = create_connection(db_name)
    if not engine:
        return
    
    try:
        ticker_listbox.delete(0, tk.END)  # Clear existing tickers
        
        query = text(f"SELECT DISTINCT ticker FROM {table_name} ORDER BY ticker")
        with engine.connect() as conn:
            result = conn.execute(query)
            tickers = [row[0] for row in result]
        
        for ticker in tickers:
            ticker_listbox.insert(tk.END, ticker)
            
        status_var.set(f"Loaded {len(tickers)} tickers from {table_name}")
    except Exception as e:
        print(f"Error loading tickers: {e}")
        messagebox.showerror("Error", f"Failed to load tickers: {str(e)}")
    finally:
        engine.dispose()

def predict_handler():
    """Handle prediction requests"""
    try:
        status_var.set("Making predictions...")
        output_text.delete(1.0, "end")

        db_name = db_var.get()
        table_name = table_var.get()
        selected_indices = ticker_listbox.curselection()
        tickers = [ticker_listbox.get(i) for i in selected_indices]
        days = days_var.get()

        if not tickers:
            error_msg = "Please select at least one ticker"
            status_var.set(error_msg)
            output_text.insert("end", f"Error: {error_msg}\n")
            messagebox.showerror("Error", error_msg)
            return

        if not trained_model:
            error_msg = "No trained model available. Please train a model first."
            status_var.set(error_msg)
            output_text.insert("end", f"Error: {error_msg}\n")
            return

        output_text.insert("end", f"=== Starting Predictions ===\nTickers: {', '.join(tickers)}\nDays: {days}\n\n")
        output_text.update()

        engine = create_connection(db_name)
        if not engine:
            error_msg = f"Could not connect to database: {db_name}"
            status_var.set(error_msg)
            output_text.insert("end", f"Error: {error_msg}\n")
            return

        try:
            placeholders = ', '.join([':ticker' + str(i) for i in range(len(tickers))])
            query = text(f"SELECT * FROM {table_name} WHERE ticker IN ({placeholders})")
            params = {f'ticker{i}': ticker for i, ticker in enumerate(tickers)}
            
            with engine.connect() as conn:
                result = conn.execute(query, params)
                df = pd.DataFrame(result.fetchall(), columns=result.keys())

            if df.empty:
                error_msg = f"No data found for selected tickers"
                status_var.set(error_msg)
                output_text.insert("end", f"Error: {error_msg}\n")
                return

            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            print(f"Prediction data rows: {len(df)}")

            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                error_msg = f"Table '{table_name}' is missing required columns: {', '.join(missing_columns)}."
                status_var.set(error_msg)
                output_text.insert("end", f"Error: {error_msg}\n")
                return

            agent = StockAIAgent()
            agent.model, agent.data_adapter.scaler = trained_model, trained_scaler
            agent.sequence_length = sequence_length_var.get()

            # Historical and future predictions for all tickers
            historical_predictions = agent.predict(df, future_only=False)
            historical_dates = df['date'].iloc[agent.sequence_length:]
            historical_predictions = historical_predictions[:len(historical_dates)]
            
            ticker_metrics = {}
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_facecolor("#3E3E3E")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_color("white")
            
            colors = matplotlib.colormaps['tab20']
            
            for i, ticker in enumerate(tickers):
                ticker_df = df[df['ticker'] == ticker]
                ticker_dates = ticker_df['date'].iloc[agent.sequence_length:]
                ticker_close = ticker_df['close'].iloc[agent.sequence_length:]
                ticker_hist_pred = historical_predictions[ticker_df.index[agent.sequence_length:]-agent.sequence_length]
                
                # Future predictions
                future_pred = agent.predict_future(ticker_df, days=days)
                last_date = pd.to_datetime(ticker_df['date'].iloc[-1])
                future_dates = [last_date + pd.Timedelta(days=j + 1) for j in range(days)]
                
                # Ensure future_pred is valid
                if future_pred is None or len(future_pred) == 0:
                    output_text.insert("end", f"Warning: No future predictions for {ticker}\n")
                    continue
                
                # Plot with color index
                color = colors(i / len(tickers))
                ax.plot(ticker_df['date'], ticker_df['close'], label=f'{ticker} Historical', color=color)
                ax.plot(ticker_dates, ticker_hist_pred, '--', label=f'{ticker} Hist. Pred.', color=color)
                ax.plot(future_dates[:len(future_pred)], future_pred, ':', label=f'{ticker} Future Pred.', color=color)
                
                # Normalization percentage (future)
                start_price = ticker_df['close'].iloc[-1]
                end_price = future_pred[-1] if future_pred else ticker_close.iloc[-1]
                norm_pct = ((end_price - start_price) / start_price) * 100
                
                # Return in dollars (assume $1,000 initial investment)
                initial_investment = 1000
                shares = initial_investment / start_price
                dollar_return = shares * (end_price - start_price)
                
                ticker_metrics[ticker] = {'norm_pct': norm_pct, 'dollar_return': dollar_return}
            
            # Rank tickers by normalization percentage
            ranked_tickers = sorted(ticker_metrics.items(), key=lambda x: x[1]['norm_pct'], reverse=True)
            
            ax.set_title("Historical and Future Predictions", color="white")
            ax.set_xlabel('Date', color="white")
            ax.set_ylabel('Price', color="white")
            ax.legend(facecolor="#2E2E2E", labelcolor="white", loc='best')
            fig.autofmt_xdate()
            canvas.draw()
            
            # Display metrics
            output_text.insert("end", "Prediction Results:\n")
            for ticker, metrics in ranked_tickers:
                output_text.insert("end", f"{ticker}:\n")
                output_text.insert("end", f"  Normalization %: {metrics['norm_pct']:.2f}%\n")
                output_text.insert("end", f"  Return ($1,000 invested): ${metrics['dollar_return']:.2f}\n")
            output_text.insert("end", "\nBest Ticker to Invest In (by Normalization %):\n")
            output_text.insert("end", f"1. {ranked_tickers[0][0]} ({ranked_tickers[0][1]['norm_pct']:.2f}%)\n")
            
            status_var.set("Predictions completed")
            output_text.insert("end", "Predictions completed successfully\n")
            
        except Exception as e:
            status_var.set("Prediction failed")
            output_text.insert("end", f"Error in prediction: {str(e)}\n")
            traceback.print_exc()
        finally:
            engine.dispose()
    except Exception as e:
        status_var.set("Prediction failed")
        output_text.insert("end", f"Error: {str(e)}\n")
        traceback.print_exc()

def main():
    """Main function to run the application"""
    databases = find_databases()
    print(f"Found databases: {databases}")
    app = initialize_gui(databases)
    app.mainloop()

if __name__ == "__main__":
    main()
