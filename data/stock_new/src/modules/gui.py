import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.config.config_manager import ConfigurationManager
from src.modules.database import DatabaseConnector
from src.modules.data_loader import DataLoader
from src.modules.stock_ai_agent import StockAIAgent
from src.modules.trading.real_trading_agent import RealTradingAgent
from src.utils.visualization import StockVisualizer

class StockGUI:
    """Main GUI interface for the Stock Market Analyzer."""
    
    def __init__(
        self,
        root: tk.Tk,
        db_connector: DatabaseConnector,
        data_adapter: DataLoader,
        ai_agent: StockAIAgent
    ):
        self.root = root
        self.db_connector = db_connector
        self.data_adapter = data_adapter
        self.ai_agent = ai_agent
        
        # Initialize trading agent as None (will be created when needed)
        self.trading_agent: Optional[RealTradingAgent] = None
        
        # Setup GUI components
        self.setup_main_frames()
        self.setup_left_panel()
        self.setup_right_panel()
        self.setup_status_bar()
    
    def setup_main_frames(self):
        """Create main frame layout."""
        # Left panel (controls)
        self.left_frame = ttk.Frame(self.root, padding="5")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Right panel (plots)
        self.right_frame = ttk.Frame(self.root, padding="5")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
    
    def setup_left_panel(self):
        """Setup all controls in the left panel."""
        # Database selection
        db_frame = ttk.LabelFrame(self.left_frame, text="Database", padding="5")
        db_frame.pack(fill=tk.X, pady=5)
        
        self.db_combo = ttk.Combobox(db_frame, state="readonly")
        self.db_combo.pack(fill=tk.X)
        self.db_combo.bind("<<ComboboxSelected>>", self.on_database_selected)
        
        # Sector selection
        sector_frame = ttk.LabelFrame(self.left_frame, text="Sector", padding="5")
        sector_frame.pack(fill=tk.X, pady=5)
        
        self.sector_combo = ttk.Combobox(sector_frame, state="readonly")
        self.sector_combo.pack(fill=tk.X)
        self.sector_combo.bind("<<ComboboxSelected>>", self.on_sector_selected)
        
        # Ticker selection
        ticker_frame = ttk.LabelFrame(self.left_frame, text="Tickers", padding="5")
        ticker_frame.pack(fill=tk.X, pady=5)
        
        self.ticker_listbox = tk.Listbox(ticker_frame, selectmode=tk.MULTIPLE)
        self.ticker_listbox.pack(fill=tk.X)
        
        # Feature selection
        feature_frame = ttk.LabelFrame(self.left_frame, text="Features", padding="5")
        feature_frame.pack(fill=tk.X, pady=5)
        
        self.feature_vars = {}
        for feature in ['close', 'volume', 'RSI', 'MACD']:
            var = tk.BooleanVar(value=True if feature == 'close' else False)
            self.feature_vars[feature] = var
            ttk.Checkbutton(
                feature_frame,
                text=feature,
                variable=var
            ).pack(anchor=tk.W)
        
        # Model configuration
        model_frame = ttk.LabelFrame(self.left_frame, text="Model", padding="5")
        model_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(model_frame, text="Type:").pack(anchor=tk.W)
        self.model_combo = ttk.Combobox(
            model_frame,
            values=['LSTM', 'GRU', 'BiLSTM', 'CNN-LSTM', 'Transformer'],
            state="readonly"
        )
        self.model_combo.set('LSTM')
        self.model_combo.pack(fill=tk.X)
        
        ttk.Label(model_frame, text="Epochs:").pack(anchor=tk.W)
        self.epochs_var = tk.StringVar(value="100")
        ttk.Entry(model_frame, textvariable=self.epochs_var).pack(fill=tk.X)
        
        # Action buttons
        button_frame = ttk.Frame(self.left_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(
            button_frame,
            text="Train Model",
            command=self.train_model_handler
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            button_frame,
            text="Predict",
            command=self.predict_handler
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            button_frame,
            text="Start Trading",
            command=self.start_trading_handler
        ).pack(fill=tk.X, pady=2)
        
        # Status text area
        self.status_text = tk.Text(self.left_frame, height=5, width=30)
        self.status_text.pack(fill=tk.X, pady=5)
    
    def setup_right_panel(self):
        """Setup plotting area in right panel."""
        # Create notebook for multiple plots
        self.plot_notebook = ttk.Notebook(self.right_frame)
        self.plot_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Historical data tab
        self.hist_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.hist_frame, text="Historical")
        
        self.hist_figure = Figure(figsize=(8, 6))
        self.hist_canvas = FigureCanvasTkAgg(self.hist_figure, self.hist_frame)
        self.hist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Training results tab
        self.train_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.train_frame, text="Training")
        
        self.train_figure = Figure(figsize=(8, 6))
        self.train_canvas = FigureCanvasTkAgg(self.train_figure, self.train_frame)
        self.train_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Predictions tab
        self.pred_frame = ttk.Frame(self.plot_notebook)
        self.plot_notebook.add(self.pred_frame, text="Predictions")
        
        self.pred_figure = Figure(figsize=(8, 6))
        self.pred_canvas = FigureCanvasTkAgg(self.pred_figure, self.pred_frame)
        self.pred_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def setup_status_bar(self):
        """Setup status bar at bottom."""
        self.status_var = tk.StringVar()
        self.status_label = ttk.Label(
            self.status_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN
        )
        self.status_label.pack(fill=tk.X)
        self.status_var.set("Ready")
    
    def on_database_selected(self, event):
        """Handle database selection."""
        db_path = self.db_combo.get()
        if db_path:
            if self.db_connector.create_connection(db_path):
                # Update sector list
                tables = self.db_connector.get_tables()
                self.sector_combo['values'] = tables
                self.status_var.set(f"Connected to {db_path}")
            else:
                messagebox.showerror("Error", f"Could not connect to {db_path}")
    
    def on_sector_selected(self, event):
        """Handle sector selection."""
        sector = self.sector_combo.get()
        if sector:
            # Update ticker list
            tickers = self.db_connector.get_unique_tickers(sector)
            self.ticker_listbox.delete(0, tk.END)
            for ticker in tickers:
                self.ticker_listbox.insert(tk.END, ticker)
            
            self.status_var.set(f"Loaded tickers for {sector}")
    
    def train_model_handler(self):
        """Handle model training."""
        try:
            # Get selected tickers
            selected_tickers = [
                self.ticker_listbox.get(idx)
                for idx in self.ticker_listbox.curselection()
            ]
            
            if not selected_tickers:
                messagebox.showwarning("Warning", "Please select at least one ticker")
                return
            
            # Get selected features
            features = [
                feat for feat, var in self.feature_vars.items()
                if var.get()
            ]
            
            if not features:
                messagebox.showwarning("Warning", "Please select at least one feature")
                return
            
            # Get model parameters
            model_type = self.model_combo.get()
            epochs = int(self.epochs_var.get())
            
            # Start training
            self.status_var.set("Training model...")
            self.status_text.insert(tk.END, f"Training {model_type} model...\n")
            
            # Get data
            sector = self.sector_combo.get()
            df = self.db_connector.load_tickers(sector)
            df = df[df['ticker'].isin(selected_tickers)]
            
            # Prepare data
            train_data = self.data_adapter.prepare_training_data(df, features)
            
            # Train model
            history = self.ai_agent.train(
                train_data.X_train,
                train_data.y_train,
                validation_data=(train_data.X_val, train_data.y_val),
                epochs=epochs
            )
            
            # Plot training results
            self.plot_training_results(history)
            
            self.status_var.set("Training completed")
            self.status_text.insert(tk.END, "Training completed\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Training error: {str(e)}")
            self.status_var.set("Training failed")
    
    def predict_handler(self):
        """Handle predictions."""
        try:
            # Get selected ticker (use first selected)
            selected_idx = self.ticker_listbox.curselection()
            if not selected_idx:
                messagebox.showwarning("Warning", "Please select a ticker")
                return
            
            ticker = self.ticker_listbox.get(selected_idx[0])
            
            # Get prediction days
            days = 30  # Could add input for this
            
            # Get data
            sector = self.sector_combo.get()
            df = self.db_connector.load_tickers(sector)
            df = df[df['ticker'] == ticker]
            
            # Make predictions
            self.status_var.set("Making predictions...")
            
            # Historical predictions
            hist_pred = self.ai_agent.predict(
                self.data_adapter.prepare_prediction_data(df)
            )
            
            # Future predictions
            future_pred = self.ai_agent.predict_future(
                self.data_adapter.prepare_prediction_data(df),
                days=days
            )
            
            # Plot predictions
            self.plot_predictions(df, hist_pred, future_pred)
            
            self.status_var.set("Predictions completed")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction error: {str(e)}")
            self.status_var.set("Prediction failed")
    
    def start_trading_handler(self):
        """Handle live trading initialization."""
        try:
            if not self.trading_agent:
                # Get API credentials (should be from secure config)
                api_key = "YOUR_API_KEY"
                api_secret = "YOUR_API_SECRET"
                
                # Initialize trading agent
                self.trading_agent = RealTradingAgent(
                    self.ai_agent,
                    api_key,
                    api_secret,
                    "https://paper-api.alpaca.markets",
                    budget=10000.0
                )
                
                self.status_var.set("Trading agent initialized")
                self.status_text.insert(tk.END, "Trading agent initialized\n")
            
            # Start trading monitoring
            self.monitor_trading()
            
        except Exception as e:
            messagebox.showerror("Error", f"Trading error: {str(e)}")
            self.status_var.set("Trading initialization failed")
    
    def monitor_trading(self):
        """Monitor trading status."""
        if self.trading_agent:
            try:
                # Get trading status
                status = self.trading_agent.monitor_trades()
                
                # Update status
                status_text = (
                    f"Equity: ${status.get('equity', 0):.2f} | "
                    f"P&L: ${status.get('total_pl', 0):.2f} | "
                    f"Positions: {status.get('open_positions', 0)}"
                )
                self.status_var.set(status_text)
                
                # Schedule next update
                self.root.after(60000, self.monitor_trading)  # Update every minute
                
            except Exception as e:
                self.status_text.insert(tk.END, f"Trading error: {str(e)}\n")
    
    def plot_training_results(self, history: Dict[str, List[float]]):
        """Plot training history."""
        self.train_figure.clear()
        ax = self.train_figure.add_subplot(111)
        
        ax.plot(history['loss'], label='Training Loss')
        ax.plot(history['val_loss'], label='Validation Loss')
        
        ax.set_title('Model Training History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        
        self.train_canvas.draw()
    
    def plot_predictions(
        self,
        df: pd.DataFrame,
        historical_pred: np.ndarray,
        future_pred: np.ndarray
    ):
        """Plot historical and future predictions."""
        self.pred_figure.clear()
        ax = self.pred_figure.add_subplot(111)
        
        # Plot historical data
        ax.plot(df['close'], label='Actual', color='blue')
        
        # Plot historical predictions
        pred_dates = df.index[len(df)-len(historical_pred):]
        ax.plot(pred_dates, historical_pred, label='Historical Predictions', color='green')
        
        # Plot future predictions
        future_dates = pd.date_range(
            start=df.index[-1],
            periods=len(future_pred)+1
        )[1:]
        ax.plot(future_dates, future_pred, label='Future Predictions', color='red')
        
        ax.set_title('Stock Price Predictions')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        
        self.pred_canvas.draw() 