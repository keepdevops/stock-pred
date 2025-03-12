"""
Control panel for the application
"""
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from config.settings import DARK_BG, LIGHT_TEXT
from data.database import get_tables, get_tickers, find_databases
import matplotlib.pyplot as plt
import os
from ui.model_viewer import ModelViewer

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
        self.view_model_btn = ttk.Button(button_frame, text="View", command=self.on_view_model)
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

    def set_load_model_callback(self, callback):
        """Set the callback for load model button"""
        self.load_model_callback = callback 

    def on_view_model(self):
        """Handler for view model button click"""
        print("View Model button clicked - direct implementation")
        model = self.model_var.get()
        print(f"Selected model: {model}")
        
        if not model:
            print("No model selected")
            from tkinter import messagebox
            messagebox.showinfo("Model Selection", "Please select a model first.")
            return
        
        # Use hardcoded models directory path
        models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
        
        # Create a simple viewer window directly here
        import os
        import time
        import tkinter as tk
        from tkinter import messagebox, ttk
        
        try:
            # Create window
            window = tk.Toplevel()
            window.title(f"Model: {model}")
            window.geometry("600x500")
            window.lift()
            window.focus_force()
            
            # Model path
            model_path = os.path.join(models_dir, model)
            
            # Create main content frame with scrolling
            main_frame = tk.Frame(window)
            main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create a canvas for scrolling
            canvas = tk.Canvas(main_frame)
            scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            self._setup_scrolling(main_frame, canvas, scrollable_frame)
            
            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Add header with model info
            header_frame = ttk.LabelFrame(scrollable_frame, text="Model Information")
            header_frame.pack(fill=tk.X, pady=10)
            
            ttk.Label(header_frame, text=f"Model: {model}", font=("Arial", 12, "bold")).pack(anchor=tk.W, padx=10, pady=5)
            ttk.Label(header_frame, text=f"Path: {model_path}").pack(anchor=tk.W, padx=10, pady=2)
            
            if os.path.exists(model_path):
                size = os.path.getsize(model_path) / (1024 * 1024)  # Convert to MB
                modified = time.ctime(os.path.getmtime(model_path))
                ttk.Label(header_frame, text=f"Size: {size:.2f} MB").pack(anchor=tk.W, padx=10, pady=2)
                ttk.Label(header_frame, text=f"Last Modified: {modified}").pack(anchor=tk.W, padx=10, pady=2)
                status_text = "Model file exists"
                status_color = "green"
            else:
                status_text = "Model file does not exist"
                status_color = "red"
            
            status_label = ttk.Label(header_frame, text=status_text, foreground=status_color)
            status_label.pack(anchor=tk.W, padx=10, pady=5)
            
            # Look for history file
            history_paths = [
                model_path.replace('.keras', '_history.pkl'),
                model_path.replace('.keras', '.history'),
                os.path.join(models_dir, model.split('.')[0] + '_history.pkl'),
                model_path + '.history',
                os.path.join(models_dir, model.split('.')[0] + '.history')
            ]
            
            history_path = None
            history_found = False
            
            for path in history_paths:
                print(f"Checking for history at: {path}")
                if os.path.exists(path):
                    history_path = path
                    history_found = True
                    print(f"History file found at: {history_path}")
                    break
            
            # Create history section
            history_frame = ttk.LabelFrame(scrollable_frame, text="Training History")
            history_frame.pack(fill=tk.X, pady=10)
            
            if history_found:
                ttk.Label(history_frame, text=f"History file: {os.path.basename(history_path)}").pack(anchor=tk.W, padx=10, pady=5)
                
                # Try to load and display history
                try:
                    import pickle
                    import matplotlib
                    matplotlib.use("TkAgg")
                    from matplotlib.figure import Figure
                    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
                    
                    with open(history_path, 'rb') as f:
                        history = pickle.load(f)
                    
                    if isinstance(history, dict):
                        print(f"History keys: {list(history.keys())}")
                        
                        # Create figure frame
                        plot_frame = ttk.Frame(history_frame)
                        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
                        
                        fig = Figure(figsize=(8, 6), dpi=100)
                        
                        # Plot loss
                        if 'loss' in history:
                            ax1 = fig.add_subplot(211)
                            ax1.plot(history['loss'], 'b-', label='Training Loss')
                            if 'val_loss' in history:
                                ax1.plot(history['val_loss'], 'r-', label='Validation Loss')
                            ax1.set_title('Model Loss')
                            ax1.set_ylabel('Loss')
                            ax1.legend()
                            ax1.grid(True)
                        
                        # Plot accuracy if available
                        if 'accuracy' in history or 'acc' in history:
                            acc_key = 'accuracy' if 'accuracy' in history else 'acc'
                            val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc' if 'val_acc' in history else None
                            
                            ax2 = fig.add_subplot(212)
                            ax2.plot(history[acc_key], 'b-', label='Training Accuracy')
                            if val_acc_key and val_acc_key in history:
                                ax2.plot(history[val_acc_key], 'r-', label='Validation Accuracy')
                            ax2.set_title('Model Accuracy')
                            ax2.set_xlabel('Epoch')
                            ax2.set_ylabel('Accuracy')
                            ax2.legend()
                            ax2.grid(True)
                        
                        fig.tight_layout()
                        
                        # Create canvas and add toolbar
                        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                        canvas.draw()
                        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                        
                        toolbar_frame = ttk.Frame(plot_frame)
                        toolbar_frame.pack(fill=tk.X)
                        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
                        toolbar.update()
                    else:
                        ttk.Label(history_frame, text="History file format not recognized",
                                 foreground="red").pack(anchor=tk.W, padx=10, pady=5)
                except Exception as e:
                    print(f"Error plotting history: {e}")
                    import traceback
                    traceback.print_exc()
                    ttk.Label(history_frame, text=f"Error plotting history: {str(e)}",
                             foreground="red").pack(anchor=tk.W, padx=10, pady=5)
            else:
                ttk.Label(history_frame, 
                         text="No training history found for this model.\n"
                              "The history file is required to display training metrics like loss and accuracy.\n"
                              "If you haven't trained this model yet, train it first to generate history.",
                         foreground="orange").pack(anchor=tk.W, padx=10, pady=10)
                
                # Add suggestion for training
                suggestion_frame = ttk.Frame(history_frame)
                suggestion_frame.pack(fill=tk.X, padx=10, pady=5)
                
                ttk.Label(suggestion_frame, 
                         text="To train a model:").pack(anchor=tk.W)
                ttk.Label(suggestion_frame, 
                         text="1. Select data source and parameters in the Training tab").pack(anchor=tk.W)
                ttk.Label(suggestion_frame, 
                         text="2. Click 'Train Model' and wait for training to complete").pack(anchor=tk.W)
                ttk.Label(suggestion_frame, 
                         text="3. Once training is complete, you can view the results here").pack(anchor=tk.W)
            
            # Add model summary section if possible
            summary_frame = ttk.LabelFrame(scrollable_frame, text="Model Architecture")
            summary_frame.pack(fill=tk.X, pady=10)
            
            try:
                if os.path.exists(model_path):
                    # Try to load model and get summary
                    try:
                        import tensorflow as tf
                        
                        model = tf.keras.models.load_model(model_path)
                        
                        # Create a StringIO to capture summary output
                        import io
                        from contextlib import redirect_stdout
                        
                        summary_str = io.StringIO()
                        with redirect_stdout(summary_str):
                            model.summary()
                        
                        # Display the summary in a text widget
                        summary_text = tk.Text(summary_frame, height=10, width=70)
                        summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
                        summary_text.insert("1.0", summary_str.getvalue())
                        summary_text.config(state="disabled")  # Make read-only
                        
                        # Add scrollbar
                        summary_scrollbar = ttk.Scrollbar(summary_frame, command=summary_text.yview)
                        summary_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                        summary_text.config(yscrollcommand=summary_scrollbar.set)
                        
                    except Exception as e:
                        print(f"Error getting model summary: {e}")
                        ttk.Label(summary_frame, 
                                 text=f"Could not load model to display architecture: {str(e)}",
                                 foreground="red").pack(anchor=tk.W, padx=10, pady=5)
                else:
                    ttk.Label(summary_frame, 
                             text="Model file does not exist. Cannot display architecture.",
                             foreground="red").pack(anchor=tk.W, padx=10, pady=5)
            except Exception as e:
                print(f"Error in summary section: {e}")
                ttk.Label(summary_frame, 
                         text=f"Error displaying model architecture: {str(e)}",
                         foreground="red").pack(anchor=tk.W, padx=10, pady=5)
            
            # Add button frame at the bottom
            button_frame = ttk.Frame(window)
            button_frame.pack(fill=tk.X, padx=10, pady=10)
            
            # Add close button
            close_btn = ttk.Button(button_frame, text="Close", command=window.destroy)
            close_btn.pack(side=tk.RIGHT, padx=5)
            
            print("Enhanced model viewer window displayed successfully")
            
        except Exception as e:
            print(f"Error in model viewer: {e}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to view model: {str(e)}")

    def set_view_model_callback(self, callback):
        """Set the callback for viewing a model"""
        self.view_model_callback = callback

    def refresh_models(self):
        """Refresh the list of trained models"""
        print("Refreshing model list")
        self.model_listbox.delete(0, tk.END)
        
        # This is a placeholder - you'll need to implement the actual model discovery
        # in your application code and provide it via a callback
        models_dir = "/Users/moose/stock-pred/data/stock-analyzer/models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        print(f"Models directory exists: {os.path.exists(models_dir)}, writable: {os.access(models_dir, os.W_OK)}")
        
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

    def _on_save_model_click(self):
        """Handler for Save Model button click with validation and feedback"""
        print(f"Save Model button clicked, ticker={self.ticker_var.get()}")
        # Get current ticker and ensure it's selected
        ticker = self.ticker_listbox.get(self.ticker_listbox.curselection()[0])
        if not ticker:
            messagebox.showwarning("Save Model", "Please select a ticker first.")
            return
        
        # Check if we have a trained model to save
        if not hasattr(self, 'current_model') or self.current_model is None:
            messagebox.showwarning("Save Model", "No trained model available to save.")
            return
        
        # Ask for model name
        model_name = simpledialog.askstring("Save Model", 
                                            f"Enter a name for the {ticker} model:",
                                            initialvalue=f"{ticker}_lstm_model")
        
        if not model_name:  # User cancelled
            return
        
        # Get model type (default to lstm if not specified)
        model_type = "lstm"  # You can make this configurable in your UI
        
        # Attempt to save the model
        success, message = self.model_controller.save_model(
            self.current_model, model_name, ticker, model_type
        )
        
        if success:
            messagebox.showinfo("Save Model", message)
            # Refresh model list if needed
            self.refresh_models()
        else:
            messagebox.showerror("Save Model Error", message) 

    def _create_model_controls(self):
        """Create the model controls without a Save Model button"""
        model_frame = ttk.LabelFrame(self, text="Model Management")
        model_frame.pack(fill=tk.X, padx=10, pady=5, ipady=5)
        
        # Model selection
        model_selection_frame = ttk.Frame(model_frame)
        model_selection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(model_selection_frame, text="Select Model:").pack(side=tk.LEFT, padx=5)
        self.model_combo = ttk.Combobox(model_selection_frame, state="readonly")
        self.model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.refresh_model_list()
        
        # Buttons for model operations
        button_frame = ttk.Frame(model_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # View Model button
        view_btn = ttk.Button(button_frame, text="View Model", command=self.on_view_model)
        view_btn.pack(side=tk.LEFT, padx=5)
        
        # Load Model button
        load_btn = ttk.Button(button_frame, text="Load Model", command=self._on_load_model_click)
        load_btn.pack(side=tk.LEFT, padx=5)
        
        # Delete Model button
        delete_btn = ttk.Button(button_frame, text="Delete Model", command=self._on_delete_model_click)
        delete_btn.pack(side=tk.LEFT, padx=5)
        
        # Note: We've removed the Save Model button
        
        # ... other buttons ... 

    def _setup_scrolling(self, container, canvas, scrollable_frame):
        """Set up scrolling for the canvas"""
        # Only apply scrollable configuration to tkinter Canvas objects, not matplotlib figures
        if isinstance(canvas, tk.Canvas):  # Check if it's a tkinter Canvas
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
        
        # Rest of your scrolling setup code... 