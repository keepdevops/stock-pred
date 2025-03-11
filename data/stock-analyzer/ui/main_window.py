"""
Main application window
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import traceback
import os
import shutil
from data.database_manager import DatabaseManager
from ui.control_panel import ControlPanel
from ui.visualization_panel import VisualizationPanel

from config.settings import WINDOW_WIDTH, WINDOW_HEIGHT, DARK_BG
from data.database import find_databases
from ui.styles import configure_styles
from ui.ui_utils import configure_dropdown_styles

class StockAnalyzerApp(tk.Tk):
    def __init__(self):
        """Initialize the main window"""
        super().__init__()
        
        print("=== Main Window Initialization Started ===")
        
        # Set up the main window
        self.title("Stock Market Analyzer")
        self.geometry("1200x800")
        print("Main window configured")
        
        # Initialize database manager and data directory
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        print(f"Data directory set to: {self.data_dir}")
        
        # Initialize database manager without the data_dir parameter
        self.database_manager = DatabaseManager()
        print("Database manager initialized")
        
        # Get available databases
        print("Attempting to retrieve available databases...")
        self.databases = self.database_manager.get_available_databases()
        print(f"Retrieved databases: {self.databases}")
        
        # Try to get tables for each database
        print("Attempting to retrieve tables for each database...")
        self.tables = {}
        for db in self.databases:
            try:
                db_tables = self.database_manager.get_tables(db)
                self.tables[db] = db_tables
                print(f"Tables in {db}: {db_tables}")
            except Exception as e:
                print(f"Error retrieving tables for {db}: {str(e)}")
        
        # Try to get tickers from tables
        print("Attempting to retrieve tickers from tables...")
        self.tickers = {}
        for db, db_tables in self.tables.items():
            self.tickers[db] = {}
            for table in db_tables:
                try:
                    table_tickers = self.database_manager.get_tickers(db, table)
                    self.tickers[db][table] = table_tickers
                    print(f"Tickers in {db}.{table}: {table_tickers}")
                except Exception as e:
                    print(f"Error retrieving tickers for {db}.{table}: {str(e)}")
        
        # Create the main frame
        self.main_frame = ttk.Frame(self)
        self.main_frame.pack(fill="both", expand=True)
        print("Main frame created")
        
        # Create the control panel and visualization panel
        print("Initializing control panel...")
        self.control_panel = ControlPanel(self.main_frame, databases=self.databases, data_dir=self.data_dir)
        self.control_panel.frame.pack(side="left", fill="y", padx=10, pady=10)
        print("Control panel initialized and packed")
        
        print("Initializing visualization panel...")
        self.visualization_panel = VisualizationPanel(self.main_frame)
        self.visualization_panel.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        print("Visualization panel initialized and packed")
        
        print("=== Main Window Initialization Completed ===")
        
        # Configure styles
        self.style = configure_styles(self)
        configure_dropdown_styles(self)
        
        # Set up event bindings
        self.control_panel.set_train_callback(self.on_train)
        self.control_panel.set_predict_callback(self.on_predict)
        self.control_panel.set_browse_db_callback(self.on_browse_database)
        self.control_panel.set_table_selected_callback(self.on_table_selected)
        
        # Global variables for trained model and scaler
        self.trained_model = None
        self.trained_scaler = None
        self.sequence_length = 10
    
    def on_train(self, params):
        """Handle training request"""
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
            
            # Validate parameters
            if not db_name:
                messagebox.showerror("Error", "Please select a database.")
                return
                
            if not table_name:
                messagebox.showerror("Error", "Please select a table.")
                return
                
            if not tickers:
                messagebox.showerror("Error", "Please select at least one ticker.")
                return
            
            # Update status
            self.control_panel.set_status("Training model...")
            
            # Create a separate thread for training to avoid UI freezing
            import threading
            
            def train_model_thread():
                try:
                    from models.lstm_model import create_lstm_model, train_model
                    from utils.data_processor import prepare_data
                    
                    # Initialize results dictionary
                    training_results = {}
                    
                    # Process each ticker
                    for ticker in tickers:
                        self.control_panel.set_status(f"Processing {ticker}...")
                        
                        # Get data from database
                        df = self.database_manager.get_data(db_name, table_name, ticker=ticker)
                        
                        if df.empty:
                            print(f"No data available for {ticker}")
                            continue
                        
                        # Prepare data for training
                        X_train, X_test, y_train, y_test, scaler = prepare_data(
                            df, sequence_length=sequence_length
                        )
                        
                        # Create model based on model type
                        if model_type == "LSTM":
                            model = create_lstm_model(
                                input_shape=(X_train.shape[1], X_train.shape[2]),
                                neurons=neurons,
                                dropout=dropout,
                                layers=layers
                            )
                        # Add other model types as needed (GRU, BiLSTM, CNN-LSTM)
                        else:
                            # Default to LSTM if model type is not recognized
                            model = create_lstm_model(
                                input_shape=(X_train.shape[1], X_train.shape[2]),
                                neurons=neurons,
                                dropout=dropout,
                                layers=layers
                            )
                        
                        # Train model
                        history = train_model(
                            model, 
                            X_train, y_train, 
                            X_test, y_test,
                            epochs=epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate
                        )
                        
                        # Store results
                        training_results[ticker] = {
                            'model': model,
                            'scaler': scaler,
                            'history': history,
                            'performance': model.evaluate(X_test, y_test)
                        }
                        
                        # Update status
                        self.control_panel.set_status(f"Trained model for {ticker}")
                    
                    # Store trained model and scaler for later use
                    self.trained_models = training_results
                    self.sequence_length = sequence_length
                    
                    # Update UI on the main thread
                    self.after(0, lambda: self._display_training_results(training_results))
                    
                except Exception as e:
                    error_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
                    print(error_msg)
                    # Update UI on the main thread
                    self.after(0, lambda: self.control_panel.set_status(f"Error: {str(e)}"))
                    self.after(0, lambda: messagebox.showerror("Training Error", error_msg))
            
            # Start training in a separate thread
            training_thread = threading.Thread(target=train_model_thread)
            training_thread.daemon = True  # Thread will close when main program exits
            training_thread.start()
            
        except Exception as e:
            error_msg = f"Error starting training: {str(e)}"
            print(error_msg)
            self.control_panel.set_status(f"Error: {str(e)}")
            messagebox.showerror("Error", error_msg)
    
    def _display_training_results(self, results):
        """Display training results in the visualization panel"""
        # Update status
        self.control_panel.set_status("Training complete")
        
        # Display results in visualization panel
        self.visualization_panel.show_training_results(results)
        
        # Show success message
        ticker_list = ", ".join(results.keys())
        messagebox.showinfo("Training Complete", f"Successfully trained models for: {ticker_list}")
        
    def on_predict(self, params):
        """Handle prediction request"""
        try:
            # Extract parameters
            db_name = params['db_name']
            table_name = params['table_name']
            tickers = params['tickers']
            days = params['days']
            
            # Validate parameters
            if not db_name:
                messagebox.showerror("Error", "Please select a database.")
                return
                
            if not table_name:
                messagebox.showerror("Error", "Please select a table.")
                return
                
            if not tickers:
                messagebox.showerror("Error", "Please select at least one ticker.")
                return
            
            # Update status
            self.control_panel.set_status(f"Showing historical data for {', '.join(tickers)}...")
            
            # Fetch historical data for all selected tickers
            for ticker in tickers:
                try:
                    # Get raw data from database
                    df = self.database_manager.get_data(db_name, table_name, ticker=ticker)
                    
                    if df.empty:
                        print(f"No data available for {ticker}")
                        continue
                    
                    # Display the historical data in the visualization panel
                    self.visualization_panel.show_historical_data(df, ticker)
                    
                except Exception as e:
                    print(f"Error fetching historical data for {ticker}: {str(e)}")
            
            # Update status
            self.control_panel.set_status("Historical data loaded")
            
        except Exception as e:
            error_msg = f"Error showing historical data: {str(e)}"
            print(error_msg)
            self.control_panel.set_status(f"Error: {str(e)}")
            messagebox.showerror("Error", error_msg)
        
    def on_browse_database(self):
        """Handle browsing for and adding databases from other directories"""
        # Open file dialog to select database files
        file_paths = filedialog.askopenfilenames(
            title="Select Database Files",
            filetypes=[
                ("Database files", "*.db *.sqlite *.sqlite3 *.duckdb"),
                ("All files", "*.*")
            ]
        )
        
        if not file_paths:
            return  # User canceled
        
        added_dbs = []
        for file_path in file_paths:
            try:
                # Get the file name without path
                file_name = os.path.basename(file_path)
                
                # Check if this is a database file
                if file_name.endswith(('.db', '.sqlite', '.sqlite3', '.duckdb')):
                    # Destination path in the data directory
                    dest_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), file_name)
                    
                    # Copy the file if it doesn't already exist
                    if not os.path.exists(dest_path):
                        shutil.copy2(file_path, dest_path)
                        added_dbs.append(file_name)
                        print(f"Added database: {file_name}")
                    else:
                        # Ask if user wants to overwrite
                        if messagebox.askyesno("File Exists", 
                                           f"The file {file_name} already exists. Overwrite?"):
                            shutil.copy2(file_path, dest_path)
                            added_dbs.append(file_name)
                            print(f"Overwritten database: {file_name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to add database {file_name}: {str(e)}")
                print(f"Error adding database {file_name}: {str(e)}")
        
        # Refresh databases if any were added
        if added_dbs:
            self.refresh_databases()
            messagebox.showinfo("Success", f"Added {len(added_dbs)} database(s):\n{', '.join(added_dbs)}")
    
    def refresh_databases(self):
        """Refresh the list of available databases"""
        # Get updated list of databases
        self.databases = self.database_manager.get_available_databases()
        print(f"Refreshed databases: {self.databases}")
        
        # Update tables for each database
        self.tables = {}
        for db in self.databases:
            try:
                db_tables = self.database_manager.get_tables(db)
                self.tables[db] = db_tables
            except Exception as e:
                print(f"Error retrieving tables for {db}: {str(e)}")
        
        # Update tickers for each table
        self.tickers = {}
        for db, db_tables in self.tables.items():
            self.tickers[db] = {}
            for table in db_tables:
                try:
                    table_tickers = self.database_manager.get_tickers(db, table)
                    self.tickers[db][table] = table_tickers
                except Exception as e:
                    print(f"Error retrieving tickers for {db}.{table}: {str(e)}")
        
        # Update the control panel with new data
        self.control_panel.refresh_db_data()
        
    def get_last_sequence(self, ticker):
        """
        Get the last sequence of data for a ticker to use for prediction
        
        Args:
            ticker: The ticker symbol to get data for
            
        Returns:
            The last sequence of normalized data
        """
        try:
            # Get the selected database and table from control panel
            db_name = self.control_panel.db_var.get()
            table_name = self.control_panel.table_var.get()
            
            if not db_name or not table_name:
                raise ValueError("Database or table not selected")
                
            # Get the data for this ticker
            df = self.database_manager.get_data(db_name, table_name, ticker=ticker)
            
            if df.empty:
                raise ValueError(f"No data available for {ticker}")
                
            # Get the sequence length used for training
            sequence_length = self.sequence_length
            
            # Log normalization process
            self.control_panel.set_status(f"Normalizing data for {ticker} prediction...")
            print(f"Preparing normalized data for {ticker} prediction")
            print(f"Data shape before normalization: {df.shape}")
            
            # Import data processor
            from utils.data_processor import prepare_data
            
            # Prepare the data (this will also normalize it)
            X_train, X_test, y_train, y_test, scaler = prepare_data(
                df, sequence_length=sequence_length
            )
            
            # Log normalization details
            scaler_type = type(scaler).__name__
            print(f"Using {scaler_type} for normalization")
            
            if hasattr(scaler, 'feature_range'):
                print(f"Normalization range: {scaler.feature_range}")
            
            # Get the last sequence from the data
            # We want the most recent data points
            if len(X_test) > 0:
                # Use the last sequence from test data if available
                last_sequence = X_test[-1]
                print(f"Using last sequence from test data for prediction")
            else:
                # Otherwise use the last sequence from train data
                last_sequence = X_train[-1]
                print(f"Using last sequence from training data for prediction")
                
            print(f"Normalized sequence shape: {last_sequence.shape}")
            
            # Update status
            self.control_panel.set_status(f"Ready to predict {ticker}")
                
            return last_sequence
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get data for prediction: {str(e)}")
            raise
    
    def on_table_selected(self, db_name, table_name, tickers):
        """Handle table selection event from control panel"""
        print(f"Table selected: {db_name}.{table_name} with {len(tickers)} tickers")
        
        try:
            # Load the tickers into the historical tab
            self.visualization_panel.load_tickers_for_historical(tickers, plot_first=False)
            
            # If tickers exist, fetch and display the first one
            if tickers:
                # Get data for the first ticker
                df = self.database_manager.get_data(db_name, table_name, ticker=tickers[0])
                
                if not df.empty:
                    # Store the data and plot it
                    self.visualization_panel.show_historical_data(df, tickers[0])
                    # Explicitly plot the data
                    self.visualization_panel._plot_historical_data(tickers[0])
                else:
                    print(f"No data found for {tickers[0]}")
        except Exception as e:
            print(f"Error loading historical data for table selection: {str(e)}")
            traceback.print_exc()
    
    def load_ticker_data(self, ticker):
        """
        Load data for a specific ticker and send it to the visualization panel
        
        Args:
            ticker: The ticker symbol to load data for
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get the current database and table
            db_name = self.control_panel.current_db.get()
            table_name = self.control_panel.current_table.get()
            
            if not db_name or not table_name:
                print(f"No database or table selected")
                return False
                
            # Get data from database using the get_data method that's used elsewhere
            print(f"Loading data for {ticker} from {db_name}.{table_name}")
            df = self.database_manager.get_data(db_name, table_name, ticker=ticker)
            
            if df is None or df.empty:
                print(f"No data found for {ticker}")
                return False
                
            # Send data to visualization panel
            self.visualization_panel.show_historical_data(df, ticker)
            return True
        except Exception as e:
            print(f"Error loading data for {ticker}: {e}")
            import traceback
            traceback.print_exc()  # Print full stack trace for debugging
            return False
    
    def run(self):
        """Run the application"""
        self.mainloop()
