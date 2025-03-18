import tkinter as tk
from tkinter import ttk, messagebox
import logging
from typing import Optional
from datetime import datetime, timedelta

from config.config_manager import ConfigurationManager
from modules.database import DatabaseConnector
from modules.data_loader import DataLoader
from modules.stock_ai_agent import StockAIAgent
from modules.trading.real_trading_agent import RealTradingAgent
from utils.visualization import StockVisualizer

class StockGUI:
    """Main GUI interface for the Stock Market Analyzer."""
    
    def __init__(
        self,
        root: tk.Tk,
        db: DatabaseConnector,
        data_loader: DataLoader,
        ai_agent: StockAIAgent,
        trading_agent: RealTradingAgent,
        config_manager: ConfigurationManager
    ):
        self.root = root
        self.db = db
        self.data_loader = data_loader
        self.ai_agent = ai_agent
        self.trading_agent = trading_agent
        self.config_manager = config_manager
        self.logger = logging.getLogger("GUI")
        
        self.visualizer = StockVisualizer()
        
        # Initialize variables
        self.selected_ticker = tk.StringVar()
        self.model_type = tk.StringVar(value="LSTM")
        self.trading_enabled = tk.BooleanVar(value=False)
        self.realtime_enabled = tk.BooleanVar(value=False)
        
        # Setup GUI components
        self.setup_gui()
    
    def setup_gui(self):
        """Initialize all GUI components."""
        # Configure root window
        self.root.title("Stock Market Analyzer")
        self.root.geometry("800x600")
        
        # Create main container
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Setup different sections
        self.setup_ticker_section()
        self.setup_data_section()
        self.setup_analysis_section()
        self.setup_trading_section()
        self.setup_status_bar()
        
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
    
    def setup_ticker_section(self):
        """Setup ticker selection section."""
        frame = ttk.LabelFrame(self.main_frame, text="Ticker Selection", padding="5")
        frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        # Ticker dropdown
        ttk.Label(frame, text="Ticker:").grid(row=0, column=0, padx=5)
        ticker_combo = ttk.Combobox(
            frame,
            textvariable=self.selected_ticker,
            values=self.config_manager.data_collection.tickers
        )
        ticker_combo.grid(row=0, column=1, padx=5)
        
        # Refresh button
        ttk.Button(
            frame,
            text="Refresh Data",
            command=self.refresh_data
        ).grid(row=0, column=2, padx=5)
    
    def setup_data_section(self):
        """Setup data management section."""
        frame = ttk.LabelFrame(self.main_frame, text="Data Management", padding="5")
        frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        # Historical data controls
        ttk.Button(
            frame,
            text="Load Historical",
            command=self.load_historical_data
        ).grid(row=0, column=0, padx=5)
        
        # Realtime toggle
        ttk.Checkbutton(
            frame,
            text="Enable Realtime",
            variable=self.realtime_enabled,
            command=self.toggle_realtime
        ).grid(row=0, column=1, padx=5)
    
    def setup_analysis_section(self):
        """Setup analysis section."""
        frame = ttk.LabelFrame(self.main_frame, text="Analysis", padding="5")
        frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        # Model selection
        ttk.Label(frame, text="Model:").grid(row=0, column=0, padx=5)
        model_combo = ttk.Combobox(
            frame,
            textvariable=self.model_type,
            values=["LSTM", "GRU", "BiLSTM", "CNN-LSTM", "Transformer"]
        )
        model_combo.grid(row=0, column=1, padx=5)
        
        # Analysis buttons
        ttk.Button(
            frame,
            text="Train Model",
            command=self.train_model
        ).grid(row=0, column=2, padx=5)
        
        ttk.Button(
            frame,
            text="Make Prediction",
            command=self.make_prediction
        ).grid(row=0, column=3, padx=5)
    
    def setup_trading_section(self):
        """Setup trading section."""
        frame = ttk.LabelFrame(self.main_frame, text="Trading", padding="5")
        frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        
        # Trading controls
        ttk.Checkbutton(
            frame,
            text="Enable Trading",
            variable=self.trading_enabled,
            command=self.toggle_trading
        ).grid(row=0, column=0, padx=5)
        
        ttk.Button(
            frame,
            text="View Positions",
            command=self.view_positions
        ).grid(row=0, column=1, padx=5)
        
        ttk.Button(
            frame,
            text="Trading History",
            command=self.view_trading_history
        ).grid(row=0, column=2, padx=5)
    
    def setup_status_bar(self):
        """Setup status bar."""
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(
            self.main_frame,
            textvariable=self.status_var,
            relief=tk.SUNKEN
        )
        status_bar.grid(row=4, column=0, sticky="ew", padx=5, pady=5)
    
    # Event handlers
    def refresh_data(self):
        """Refresh data for selected ticker."""
        ticker = self.selected_ticker.get()
        if not ticker:
            messagebox.showwarning("Warning", "Please select a ticker")
            return
        
        try:
            self.status_var.set(f"Refreshing data for {ticker}...")
            self.data_loader.refresh_ticker_data(ticker)
            self.status_var.set("Data refresh complete")
        except Exception as e:
            self.logger.error(f"Error refreshing data: {str(e)}")
            messagebox.showerror("Error", f"Failed to refresh data: {str(e)}")
            self.status_var.set("Ready")
    
    def load_historical_data(self):
        """Load historical data for selected ticker."""
        ticker = self.selected_ticker.get()
        if not ticker:
            messagebox.showwarning("Warning", "Please select a ticker")
            return
        
        try:
            self.status_var.set(f"Loading historical data for {ticker}...")
            self.data_loader.collect_historical_data(ticker)
            self.status_var.set("Historical data loaded")
        except Exception as e:
            self.logger.error(f"Error loading historical data: {str(e)}")
            messagebox.showerror("Error", f"Failed to load historical data: {str(e)}")
            self.status_var.set("Ready")
    
    def toggle_realtime(self):
        """Toggle realtime data collection."""
        if self.realtime_enabled.get():
            self.start_realtime_collection()
        else:
            self.stop_realtime_collection()
    
    def train_model(self):
        """Train model for selected ticker."""
        ticker = self.selected_ticker.get()
        if not ticker:
            messagebox.showwarning("Warning", "Please select a ticker")
            return
        
        try:
            self.status_var.set(f"Training model for {ticker}...")
            metrics = self.ai_agent.train_model(
                ticker,
                model_type=self.model_type.get()
            )
            self.status_var.set("Model training complete")
            messagebox.showinfo("Training Complete", f"Training metrics: {metrics}")
        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            messagebox.showerror("Error", f"Failed to train model: {str(e)}")
            self.status_var.set("Ready")
    
    def make_prediction(self):
        """Make prediction for selected ticker."""
        ticker = self.selected_ticker.get()
        if not ticker:
            messagebox.showwarning("Warning", "Please select a ticker")
            return
        
        try:
            self.status_var.set(f"Making prediction for {ticker}...")
            predictions = self.ai_agent.make_prediction(ticker)
            self.status_var.set("Prediction complete")
            messagebox.showinfo("Prediction", f"Predictions: {predictions}")
        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            messagebox.showerror("Error", f"Failed to make prediction: {str(e)}")
            self.status_var.set("Ready")
    
    def toggle_trading(self):
        """Toggle trading system."""
        if self.trading_enabled.get():
            self.trading_agent.start_trading()
            self.status_var.set("Trading system started")
        else:
            self.trading_agent.stop_trading()
            self.status_var.set("Trading system stopped")
    
    def view_positions(self):
        """View current trading positions."""
        try:
            positions = self.trading_agent.get_position_summary()
            if positions.empty:
                messagebox.showinfo("Positions", "No open positions")
            else:
                # Create a new window to display positions
                self.show_dataframe(positions, "Current Positions")
        except Exception as e:
            self.logger.error(f"Error viewing positions: {str(e)}")
            messagebox.showerror("Error", f"Failed to view positions: {str(e)}")
    
    def view_trading_history(self):
        """View trading history."""
        try:
            history = self.trading_agent.get_trade_history()
            if history.empty:
                messagebox.showinfo("History", "No trading history")
            else:
                # Create a new window to display history
                self.show_dataframe(history, "Trading History")
        except Exception as e:
            self.logger.error(f"Error viewing history: {str(e)}")
            messagebox.showerror("Error", f"Failed to view history: {str(e)}")
    
    def show_dataframe(self, df, title):
        """Display a DataFrame in a new window."""
        window = tk.Toplevel(self.root)
        window.title(title)
        
        # Create treeview
        tree = ttk.Treeview(window)
        tree["columns"] = list(df.columns)
        tree["show"] = "headings"
        
        # Set column headings
        for column in df.columns:
            tree.heading(column, text=column)
            tree.column(column, width=100)
        
        # Add data
        for i, row in df.iterrows():
            tree.insert("", "end", values=list(row))
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(window, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack widgets
        tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y") 