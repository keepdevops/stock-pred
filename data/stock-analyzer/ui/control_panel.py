"""
Control panel for the application
"""
import tkinter as tk
from tkinter import ttk, messagebox
from config.settings import DARK_BG, LIGHT_TEXT
from data.database import get_tables, get_tickers, find_databases
import matplotlib.pyplot as plt

class ControlPanel:
    def __init__(self, parent, databases, data_dir=None):
        self.parent = parent
        self.databases = databases
        self.data_dir = data_dir
        self.train_callback = None
        self.predict_callback = None
        self.browse_db_callback = None
        self.table_selected_callback = None
        self.save_model_callback = None
        self.load_model_callback = None
        self.view_model_callback = None
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="Control Panel")
        self.frame.pack(side="left", fill="y", padx=5, pady=5)
        
        # Database selection
        self.create_database_section()
        
        # Model configuration
        self.create_model_section()
        
        # Training configuration
        self.create_training_section()
        
        # Prediction configuration
        self.create_prediction_section()
        
        # Training Results
        self.create_training_results_section()
        
        # Action buttons
        self.create_action_section()
        
        # Initial refreshes
        self.refresh_models()  # Add this line to populate models on startup
        
    def create_database_section(self):
        """Create database selection section"""
        db_frame = ttk.LabelFrame(self.frame, text="Database")
        db_frame.pack(fill="x", padx=5, pady=5)
        
        # Database selection
        ttk.Label(db_frame, text="Database:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.db_var = tk.StringVar()
        self.db_combo = ttk.Combobox(db_frame, textvariable=self.db_var, state="readonly")
        self.db_combo['values'] = self.databases
        if self.databases:
            self.db_combo.current(0)
        self.db_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.db_combo.bind("<<ComboboxSelected>>", self.on_db_selected)
        
        # Database action buttons
        db_buttons_frame = ttk.Frame(db_frame)
        db_buttons_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=2)
        
        # Add DB button
        self.browse_db_btn = ttk.Button(db_buttons_frame, text="Browse for DB", command=self.on_browse_db_clicked)
        self.browse_db_btn.pack(side="left", fill="x", expand=True, padx=2, pady=2)
        
        # Refresh button
        refresh_btn = ttk.Button(db_buttons_frame, text="Refresh", command=self.refresh_db_data)
        refresh_btn.pack(side="right", fill="x", expand=True, padx=2, pady=2)
        
        # Table selection
        ttk.Label(db_frame, text="Table:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.table_var = tk.StringVar()
        self.table_combo = ttk.Combobox(db_frame, textvariable=self.table_var, state="readonly")
        self.table_combo.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        self.table_combo.bind("<<ComboboxSelected>>", self.on_table_selected)
        
        # Ticker selection
        ttk.Label(db_frame, text="Tickers:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        
        # Create a frame for the listbox and scrollbar
        ticker_frame = ttk.Frame(db_frame)
        ticker_frame.grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(ticker_frame)
        scrollbar.pack(side="right", fill="y")
        
        # Create listbox with multiple selection
        self.ticker_listbox = tk.Listbox(ticker_frame, selectmode="multiple", 
                                         yscrollcommand=scrollbar.set,
                                         bg="white", fg="black", height=5)
        self.ticker_listbox.pack(side="left", fill="both", expand=True)
        
        # Configure scrollbar
        scrollbar.config(command=self.ticker_listbox.yview)
        
        # Add Select All and Clear All buttons
        ticker_buttons_frame = ttk.Frame(db_frame)
        ticker_buttons_frame.grid(row=4, column=1, sticky="ew", padx=5, pady=2)
        
        # Select All button
        select_all_btn = ttk.Button(ticker_buttons_frame, text="Select All", command=self.select_all_tickers)
        select_all_btn.pack(side="left", fill="x", expand=True, padx=2, pady=2)
        
        # Clear All button
        clear_all_btn = ttk.Button(ticker_buttons_frame, text="Clear All", command=self.clear_all_tickers)
        clear_all_btn.pack(side="right", fill="x", expand=True, padx=2, pady=2)
        
    def create_model_section(self):
        """Create model configuration section"""
        model_frame = ttk.LabelFrame(self.frame, text="Model Configuration")
        model_frame.pack(fill="x", padx=5, pady=5)
        
        # Model type
        ttk.Label(model_frame, text="Model Type:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.model_var = tk.StringVar(value="LSTM")
        model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly")
        model_combo['values'] = ["LSTM", "GRU", "BiLSTM", "CNN-LSTM"]
        model_combo.current(0)
        model_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        
        # Sequence length
        ttk.Label(model_frame, text="Sequence Length:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.seq_var = tk.IntVar(value=10)
        seq_spin = ttk.Spinbox(model_frame, from_=5, to=50, textvariable=self.seq_var, style="BlackText.TSpinbox")
        seq_spin.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        
        # Neurons
        ttk.Label(model_frame, text="Neurons:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.neurons_var = tk.IntVar(value=50)
        neurons_spin = ttk.Spinbox(model_frame, from_=10, to=200, textvariable=self.neurons_var, style="BlackText.TSpinbox")
        neurons_spin.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        
        # Layers
        ttk.Label(model_frame, text="Layers:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.layers_var = tk.IntVar(value=2)
        layers_spin = ttk.Spinbox(model_frame, from_=1, to=5, textvariable=self.layers_var, style="BlackText.TSpinbox")
        layers_spin.grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        
        # Dropout
        ttk.Label(model_frame, text="Dropout:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        self.dropout_var = tk.DoubleVar(value=0.2)
        dropout_spin = ttk.Spinbox(model_frame, from_=0.0, to=0.5, increment=0.1, 
                                  textvariable=self.dropout_var, style="BlackText.TSpinbox")
        dropout_spin.grid(row=4, column=1, sticky="ew", padx=5, pady=2)
        
    def create_training_section(self):
        """Create training configuration section"""
        train_frame = ttk.LabelFrame(self.frame, text="Training Configuration")
        train_frame.pack(fill="x", padx=5, pady=5)
        
        # Epochs
        ttk.Label(train_frame, text="Epochs:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.epochs_var = tk.IntVar(value=50)
        epochs_spin = ttk.Spinbox(train_frame, from_=10, to=500, textvariable=self.epochs_var, style="BlackText.TSpinbox")
        epochs_spin.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        
        # Batch size
        ttk.Label(train_frame, text="Batch Size:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.batch_var = tk.IntVar(value=32)
        batch_spin = ttk.Spinbox(train_frame, from_=8, to=128, textvariable=self.batch_var, style="BlackText.TSpinbox")
        batch_spin.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        
        # Learning rate
        ttk.Label(train_frame, text="Learning Rate:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.lr_var = tk.DoubleVar(value=0.001)
        lr_spin = ttk.Spinbox(train_frame, from_=0.0001, to=0.01, increment=0.0001, 
                             textvariable=self.lr_var, style="BlackText.TSpinbox")
        lr_spin.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        
    def create_prediction_section(self):
        """Create prediction configuration section"""
        pred_frame = ttk.LabelFrame(self.frame, text="Prediction Configuration")
        pred_frame.pack(fill="x", padx=5, pady=5)
        
        # Days to predict
        ttk.Label(pred_frame, text="Days to Predict:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.days_var = tk.IntVar(value=30)
        days_spin = ttk.Spinbox(pred_frame, from_=1, to=365, textvariable=self.days_var, style="BlackText.TSpinbox")
        days_spin.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        
    def create_training_results_section(self):
        """Create training results section with model listbox"""
        results_frame = ttk.LabelFrame(self.frame, text="Training Results")
        results_frame.pack(fill="x", padx=5, pady=5)
        
        # Create a frame for the listbox and scrollbar
        model_frame = ttk.Frame(results_frame)
        model_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(model_frame)
        scrollbar.pack(side="right", fill="y")
        
        # Label for trained models
        ttk.Label(results_frame, text="Trained Models:").pack(anchor="w", padx=5, pady=2)
        
        # Create listbox
        self.model_listbox = tk.Listbox(model_frame, 
                                        yscrollcommand=scrollbar.set,
                                        bg="white", fg="black", height=4)
        self.model_listbox.pack(side="left", fill="both", expand=True)
        
        # Configure scrollbar
        scrollbar.config(command=self.model_listbox.yview)
        
        # Button frame
        button_frame = ttk.Frame(results_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        # View button
        self.view_model_btn = ttk.Button(button_frame, text="View", command=self.on_view_model_clicked)
        self.view_model_btn.pack(side="left", fill="x", expand=True, padx=2, pady=2)
        
        # Refresh button
        refresh_models_btn = ttk.Button(button_frame, text="Refresh", command=self.refresh_models)
        refresh_models_btn.pack(side="right", fill="x", expand=True, padx=2, pady=2)
        
    def create_action_section(self):
        """Create action buttons section"""
        action_frame = ttk.Frame(self.frame)
        action_frame.pack(fill="x", padx=5, pady=10)
        
        # Train button
        self.train_btn = ttk.Button(action_frame, text="Train Model", command=self.on_train_clicked)
        self.train_btn.pack(fill="x", pady=2)
        
        # Save model button
        self.save_model_btn = ttk.Button(action_frame, text="Save Model", command=self.on_save_model_clicked)
        self.save_model_btn.pack(fill="x", pady=2)
        
        # Load model button
        self.load_model_btn = ttk.Button(action_frame, text="Load Model", command=self.on_load_model_clicked)
        self.load_model_btn.pack(fill="x", pady=2)
        
        # Predict button
        self.predict_btn = ttk.Button(action_frame, text="Make Predictions", command=self.on_predict_clicked)
        self.predict_btn.pack(fill="x", pady=2)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(action_frame, textvariable=self.status_var)
        status_label.pack(fill="x", pady=5)
        
    def on_db_selected(self, event):
        """Handle database selection"""
        db_name = self.db_var.get()
        tables = get_tables(db_name, self.data_dir)
        self.table_combo['values'] = tables
        if tables:
            self.table_combo.current(0)
            self.on_table_selected(None)
        else:
            self.table_var.set("")
            self.ticker_listbox.delete(0, tk.END)
            
    def on_table_selected(self, event):
        """Handle table selection"""
        db_name = self.db_var.get()
        table_name = self.table_var.get()
        tickers = get_tickers(db_name, table_name, self.data_dir)
        
        self.ticker_listbox.delete(0, tk.END)
        for ticker in tickers:
            self.ticker_listbox.insert(tk.END, ticker)
            
        # Call the table selected callback if set
        if self.table_selected_callback:
            self.table_selected_callback(db_name, table_name, tickers)
        
    def refresh_db_data(self):
        """Refresh database data"""
        self.databases = find_databases(self.data_dir)
        self.db_combo['values'] = self.databases
        if self.db_var.get() not in self.databases and self.databases:
            self.db_combo.current(0)
            self.on_db_selected(None)
    
    def on_browse_db_clicked(self):
        """Handle browse for database button click"""
        if self.browse_db_callback:
            self.browse_db_callback()
            
    def on_train_clicked(self):
        """Handle train button click"""
        if self.train_callback:
            params = {
                'db_name': self.db_var.get(),
                'table_name': self.table_var.get(),
                'tickers': [self.ticker_listbox.get(i) for i in self.ticker_listbox.curselection()],
                'model_type': self.model_var.get(),
                'sequence_length': self.seq_var.get(),
                'neurons': self.neurons_var.get(),
                'layers': self.layers_var.get(),
                'dropout': self.dropout_var.get(),
                'epochs': self.epochs_var.get(),
                'batch_size': self.batch_var.get(),
                'learning_rate': self.lr_var.get()
            }
            self.train_callback(params)
            
    def on_predict_clicked(self):
        """Handle predict button click"""
        if self.predict_callback:
            params = {
                'db_name': self.db_var.get(),
                'table_name': self.table_var.get(),
                'tickers': [self.ticker_listbox.get(i) for i in self.ticker_listbox.curselection()],
                'days': self.days_var.get()
            }
            self.predict_callback(params)
            
    def on_save_model_clicked(self):
        """Handle save model button click"""
        print("Save Model button clicked")
        if self.save_model_callback:
            # Show acknowledgment that model is being saved to the models directory
            self.set_status("Saving model to 'models' directory...")
            print("Saving model to 'models' directory...")
            messagebox.showinfo("Saving Model", "Model will be saved to the 'models' directory")
            
            try:
                success = self.save_model_callback()
                if success:
                    print("Model saved successfully to 'models' directory")
                    messagebox.showinfo("Success", "Model saved successfully to 'models' directory!")
                    self.set_status("Model saved successfully to 'models' directory")
                else:
                    print("Failed to save model to 'models' directory")
                    messagebox.showerror("Error", "Failed to save model to 'models' directory")
                    self.set_status("Failed to save model to 'models' directory")
            except Exception as e:
                print(f"Error while saving model: {str(e)}")
                messagebox.showerror("Error", f"An error occurred while saving the model: {str(e)}")
                self.set_status(f"Error: {str(e)}")
        
    def on_load_model_clicked(self):
        """Handle load model button click"""
        print("Load Model button clicked")
        if self.load_model_callback:
            self.set_status("Loading model...")
            print("Loading model...")
            try:
                success = self.load_model_callback()
                if success:
                    print("Model loaded successfully")
                    messagebox.showinfo("Success", "Model loaded successfully!")
                    self.set_status("Model loaded successfully")
                else:
                    print("Failed to load model")
                    messagebox.showerror("Error", "Failed to load model")
                    self.set_status("Failed to load model")
            except Exception as e:
                print(f"Error while loading model: {str(e)}")
                messagebox.showerror("Error", f"An error occurred while loading the model: {str(e)}")
                self.set_status(f"Error: {str(e)}")
            
    def set_train_callback(self, callback):
        """Set the callback for train button"""
        self.train_callback = callback
        
    def set_predict_callback(self, callback):
        """Set the callback for predict button"""
        self.predict_callback = callback
        
    def set_browse_db_callback(self, callback):
        """Set the callback for browse database button"""
        self.browse_db_callback = callback
        
    def set_status(self, status):
        """Set the status text"""
        self.status_var.set(status)

    def select_all_tickers(self):
        """Select all tickers in the listbox"""
        self.ticker_listbox.selection_set(0, tk.END)
        
    def clear_all_tickers(self):
        """Clear all ticker selections in the listbox"""
        self.ticker_listbox.selection_clear(0, tk.END)

    def get_last_sequence(self, ticker):
        """
        Get the last sequence of data for a ticker to use for prediction
        """ 

    def set_table_selected_callback(self, callback):
        """Set the callback for table selection"""
        self.table_selected_callback = callback 

    @property
    def current_db(self):
        """Get the currently selected database"""
        return self.db_var
        
    @property
    def current_table(self):
        """Get the currently selected table"""
        return self.table_var 

    def set_save_model_callback(self, callback):
        """Set the callback for save model button"""
        self.save_model_callback = callback
        
    def set_load_model_callback(self, callback):
        """Set the callback for load model button"""
        self.load_model_callback = callback 

    def on_view_model_clicked(self):
        """Handle view model button click"""
        print("View Model button clicked")
        selected_indices = self.model_listbox.curselection()
        
        if not selected_indices:
            messagebox.showinfo("Information", "Please select a model from the list")
            return
        
        selected_model = self.model_listbox.get(selected_indices[0])
        print(f"Selected model: {selected_model}")
        
        if self.view_model_callback:
            self.set_status(f"Viewing model: {selected_model}")
            self.view_model_callback(selected_model)

    def set_view_model_callback(self, callback):
        """Set the callback for viewing a model"""
        self.view_model_callback = callback

    def refresh_models(self):
        """Refresh the list of trained models"""
        print("Refreshing model list")
        self.model_listbox.delete(0, tk.END)
        
        # This is a placeholder - you'll need to implement the actual model discovery
        # in your application code and provide it via a callback
        import os
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5') or f.endswith('.keras')]
            for model_file in model_files:
                self.model_listbox.insert(tk.END, model_file)
        
        if self.model_listbox.size() == 0:
            self.model_listbox.insert(tk.END, "No models found")
            self.view_model_btn.config(state="disabled")
        else:
            self.view_model_btn.config(state="normal")
        
        self.set_status("Model list refreshed") 