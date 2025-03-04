import os
# Set environment variable to silence Tk deprecation warning
os.environ['TK_SILENCE_DEPRECATION'] = '1'  # Silence Tk deprecation warning

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
import tkinter as tk
from tkinter import ttk, messagebox
import traceback
from datetime import datetime
import glob
from sklearn.preprocessing import MinMaxScaler
import pickle

# Set Matplotlib backend
plt.switch_backend('TkAgg')

# Global variables for trained model and scaler
trained_model = None
trained_scaler = None

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
            df = df[self.features].fillna(method='ffill').fillna(method='bfill').fillna(0)
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
            df = df.set_index('date').sort_index()

            df = self.calculate_technical_indicators(df)
            df = df[self.features].fillna(method='ffill').fillna(method='bfill').fillna(0)
            scaled_data = self.scaler.transform(df)

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
            return df.fillna(method='ffill').fillna(method='bfill').fillna(0)
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

    def predict(self, df):
        """Make predictions using the trained model"""
        try:
            print("\nStarting prediction...")
            if self.model is None:
                raise ValueError("Model not trained")

            X = self.data_adapter.prepare_prediction_data(df)
            if X is None:
                raise ValueError("Prediction data preparation failed")

            predictions = self.model.predict(X)
            return self.data_adapter.inverse_transform_predictions(predictions)
        except Exception as e:
            print(f"Error making predictions: {e}")
            traceback.print_exc()
            return None

def initialize_control_panel(main_frame, databases):
    """Initialize the control panel with database, table, and ticker controls"""
    try:
        print("Creating control panel...")
        control_panel = ttk.Frame(main_frame)
        control_panel.pack(side="left", fill="y", padx=10, pady=10)

        # Create a frame that will contain the canvas and scrollbar
        control_container = ttk.Frame(control_panel)
        control_container.pack(side="top", fill="both", expand=True)
        
        # Create canvas with a fixed width that will hold all controls
        control_canvas = tk.Canvas(control_container, width=280)
        control_scrollbar = ttk.Scrollbar(control_container, orient="vertical", command=control_canvas.yview)
        
        # Create the scrollable frame that will contain all the controls
        scrollable_control_frame = ttk.Frame(control_canvas)
        scrollable_control_frame.bind(
            "<Configure>", 
            lambda e: control_canvas.configure(scrollregion=control_canvas.bbox("all"))
        )
        
        # Make the scrollable frame expand to fill the canvas width
        def _configure_frame(event):
            # Update the scrollbar to match the size of the inner frame
            size = (scrollable_control_frame.winfo_reqwidth(), scrollable_control_frame.winfo_reqheight())
            control_canvas.config(scrollregion="0 0 %s %s" % size)
            # Make sure the frame fills the canvas width
            if scrollable_control_frame.winfo_reqwidth() != control_canvas.winfo_width():
                control_canvas.config(width=scrollable_control_frame.winfo_reqwidth())
        
        control_canvas.bind("<Configure>", _configure_frame)
        
        # Create window inside canvas which will be scrolled
        control_canvas.create_window((0, 0), window=scrollable_control_frame, anchor="nw")
        
        # Pack the scrollbar and canvas properly
        control_scrollbar.pack(side="right", fill="y")
        control_canvas.pack(side="left", fill="both", expand=True)
        
        # Configure the canvas to respond to scrollbar
        control_canvas.configure(yscrollcommand=control_scrollbar.set)
        
        # Add mousewheel scrolling support
        def _on_mousewheel(event):
            control_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        control_canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows/macOS
        control_canvas.bind_all("<Button-4>", lambda e: control_canvas.yview_scroll(-1, "units"))  # Linux
        control_canvas.bind_all("<Button-5>", lambda e: control_canvas.yview_scroll(1, "units"))   # Linux

        status_var = tk.StringVar(value="Ready")
        db_var = tk.StringVar()
        table_var = tk.StringVar()
        days_var = tk.IntVar(value=30)
        epochs_var = tk.IntVar(value=50)
        batch_size_var = tk.IntVar(value=32)
        learning_rate_var = tk.DoubleVar(value=0.001)
        sequence_length_var = tk.IntVar(value=10)

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
        ticker_listbox = tk.Listbox(ticker_frame, selectmode="multiple", height=6, bg="#FFFFFF", fg="black", width=30)
        ticker_scrollbar = ttk.Scrollbar(ticker_frame, orient="vertical", command=ticker_listbox.yview)
        ticker_listbox.configure(yscrollcommand=ticker_scrollbar.set)
        ticker_scrollbar.pack(side="right", fill="y")
        ticker_listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        # Add selection control buttons
        ticker_buttons_frame = ttk.Frame(ticker_frame)
        ticker_buttons_frame.pack(side="bottom", fill="x", padx=5, pady=2)

        # Select All button
        def select_all_tickers():
            ticker_listbox.select_set(0, tk.END)
        select_all_button = ttk.Button(ticker_buttons_frame, text="Select All", command=select_all_tickers, width=13)
        select_all_button.pack(side="left", padx=2, pady=2)

        # Clear All button
        def clear_all_tickers():
            ticker_listbox.selection_clear(0, tk.END)
        clear_all_button = ttk.Button(ticker_buttons_frame, text="Clear All", command=clear_all_tickers, width=13)
        clear_all_button.pack(side="right", padx=2, pady=2)

        # Duration controls
        duration_frame = ttk.LabelFrame(scrollable_control_frame, text="Prediction Duration")
        duration_frame.pack(fill="x", padx=5, pady=5)
        days_combo = ttk.Combobox(duration_frame, textvariable=days_var, values=[1, 7, 14, 30, 60, 90], state="readonly", width=28)
        days_combo.pack(fill="x", padx=5, pady=5)

        # AI controls
        ai_frame = ttk.LabelFrame(scrollable_control_frame, text="AI Controls")
        ai_frame.pack(fill="x", padx=5, pady=5)
        ttk.Label(ai_frame, text="Epochs:").pack(anchor="w", padx=5)
        epochs_entry = ttk.Entry(ai_frame, textvariable=epochs_var, width=28, style="EntryStyle.TEntry")
        epochs_entry.pack(fill="x", padx=5, pady=2)
        ttk.Label(ai_frame, text="Batch Size:").pack(anchor="w", padx=5)
        batch_size_entry = ttk.Entry(ai_frame, textvariable=batch_size_var, width=28, style="EntryStyle.TEntry")
        batch_size_entry.pack(fill="x", padx=5, pady=2)
        ttk.Label(ai_frame, text="Learning Rate:").pack(anchor="w", padx=5)
        learning_rate_entry = ttk.Entry(ai_frame, textvariable=learning_rate_var, width=28, style="EntryStyle.TEntry")
        learning_rate_entry.pack(fill="x", padx=5, pady=2)
        ttk.Label(ai_frame, text="Sequence Length:").pack(anchor="w", padx=5)
        sequence_length_entry = ttk.Entry(ai_frame, textvariable=sequence_length_var, width=28, style="EntryStyle.TEntry")
        sequence_length_entry.pack(fill="x", padx=5, pady=2)

        # Buttons
        buttons_frame = ttk.Frame(ai_frame)
        buttons_frame.pack(fill="x", padx=5, pady=5)
        train_button = ttk.Button(buttons_frame, text="Train Model")
        train_button.pack(side="left", fill="x", expand=True, padx=2)
        predict_button = ttk.Button(buttons_frame, text="Make Prediction")
        predict_button.pack(side="right", fill="x", expand=True, padx=2)

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
            engine = create_connection(db_name)
            if engine:
                print(f"Database selected: {db_name}")
                try:
                    with engine.connect() as conn:
                        # Use text() to create an executable SQL statement
                        tables = conn.execute(text("SHOW TABLES")).fetchall()
                    tables = [t[0] for t in tables]
                    table_combo["values"] = tables
                    if tables:
                        table_var.set(tables[0])
                        on_table_selected()
                except Exception as e:
                    print(f"Error fetching tables: {e}")
                    # Fallback method for SQLite databases
                    with engine.connect() as conn:
                        tables = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
                    tables = [t[0] for t in tables]
                    table_combo["values"] = tables
                    if tables:
                        table_var.set(tables[0])
                        on_table_selected()
                finally:
                    engine.dispose()

        def on_table_selected(event=None):
            db_name = db_var.get()
            table_name = table_var.get()
            if not db_name or not table_name:
                return
            print(f"Table selected: {table_name}")
            engine = create_connection(db_name)
            if engine:
                ticker_listbox.delete(0, tk.END)
                with engine.connect() as conn:
                    # Use text() to create an executable SQL statement and parameters binding
                    query = text(f"SELECT DISTINCT ticker FROM {table_name} ORDER BY ticker")
                    tickers = conn.execute(query).fetchall()
                tickers = [t[0] for t in tickers]
                for ticker in tickers:
                    ticker_listbox.insert(tk.END, ticker)
                if tickers:
                    ticker_listbox.selection_set(0)
                engine.dispose()

        def train_model_handler():
            global trained_model, trained_scaler
            status_var.set("Training model...")
            output_text.delete(1.0, "end")

            db_name = db_var.get()
            table_name = table_var.get()
            selected_indices = ticker_listbox.curselection()
            tickers = [ticker_listbox.get(i) for i in selected_indices]

            if not tickers:
                error_msg = "Please select at least one ticker before training"
                status_var.set(error_msg)
                output_text.insert("end", f"Error: {error_msg}\n")
                messagebox.showerror("Error", error_msg)
                return

            ticker = tickers[0]
            output_text.insert("end", f"=== Starting Model Training ===\nDatabase: {db_name}\nTable: {table_name}\nTicker: {ticker}\n\n")
            print(f"\n=== Starting Model Training ===\nDatabase: {db_name}\nTable: {table_name}\nTicker: {ticker}")

            engine = create_connection(db_name)
            query = text(f"SELECT * FROM {table_name} WHERE ticker = :ticker")
            df = pd.read_sql_query(query, engine, params={"ticker": ticker})
            engine.dispose()

            if df.empty:
                error_msg = f"No data found for ticker {ticker}"
                status_var.set(error_msg)
                output_text.insert("end", f"Error: {error_msg}\n")
                return

            agent = StockAIAgent()
            input_shape = (sequence_length_var.get(), len(agent.data_adapter.features))
            agent.build_model(input_shape)
            history = agent.train(df, epochs=epochs_var.get(), batch_size=batch_size_var.get())

            if history:
                save_model(agent.model, agent.data_adapter.scaler, ticker)
                trained_model, trained_scaler = agent.model, agent.data_adapter.scaler
                status_var.set("Model training completed")
                output_text.insert("end", "Model training completed successfully\n")
            else:
                status_var.set("Model training failed")
                output_text.insert("end", "Model training failed\n")

        def predict_handler():
            global trained_model, trained_scaler
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

            ticker = tickers[0]
            output_text.insert("end", f"=== Starting Predictions ===\nTicker: {ticker}\nDays: {days}\n\n")
            engine = create_connection(db_name)
            query = text(f"SELECT * FROM {table_name} WHERE ticker = :ticker ORDER BY date")
            df = pd.read_sql_query(query, engine, params={"ticker": ticker})
            engine.dispose()

            agent = StockAIAgent()
            agent.model, agent.data_adapter.scaler = trained_model, trained_scaler
            predictions = agent.predict(df)

            if predictions is not None:
                last_date = pd.to_datetime(df['date'].iloc[-1])
                future_dates = [last_date + pd.Timedelta(days=i + 1) for i in range(days)]
                fig.clear()
                ax = fig.add_subplot(111)
                ax.plot(df['date'], df['close'], label='Historical')
                ax.plot(future_dates[:len(predictions)], predictions, 'r--', label='Predicted')
                ax.set_title(f"{ticker} Prediction")
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.legend()
                fig.autofmt_xdate()
                canvas.draw()
                status_var.set("Predictions completed")
                output_text.insert("end", "Predictions completed successfully\n")
            else:
                status_var.set("Prediction failed")
                output_text.insert("end", "Prediction failed\n")

        db_combo.bind("<<ComboboxSelected>>", on_database_selected)
        table_combo.bind("<<ComboboxSelected>>", on_table_selected)
        train_button.configure(command=train_model_handler)
        predict_button.configure(command=predict_handler)

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
        style.configure("EntryStyle.TEntry", foreground='black', fieldbackground='white')
        
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
        model.save(os.path.join(model_path, f"{ticker}_model.h5"))
        with open(os.path.join(model_path, f"{ticker}_scaler.pkl"), 'wb') as f:
            pickle.dump(scaler, f)
        print(f"Model and scaler saved for {ticker}")
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        return False

def main():
    """Main function to run the application"""
    databases = find_databases()
    print(f"Found databases: {databases}")
    app = initialize_gui(databases)
    app.mainloop()

if __name__ == "__main__":
    main()
