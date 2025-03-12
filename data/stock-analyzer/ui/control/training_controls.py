"""
Training controls panel
"""
import tkinter as tk
import os
from tkinter import ttk, messagebox, filedialog
from ui.control.base_panel import BasePanel

class TrainingControls(BasePanel):
    def __init__(self, parent, event_bus):
        super().__init__(parent, event_bus, "Training Configuration")
        
        # Create UI components
        self._create_ui()
        
        # Subscribe to events
        self.event_bus.subscribe("model_saved", self._on_model_saved)
        self.event_bus.subscribe("model_loaded", self._on_model_loaded)
        self.event_bus.subscribe("model_save_error", self._on_model_save_error)
        self.event_bus.subscribe("model_load_error", self._on_model_load_error)
        
    def _create_ui(self):
        """Create the UI components"""
        # Model configuration
        model_frame = ttk.Frame(self)
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
        seq_spin = ttk.Spinbox(model_frame, from_=5, to=50, textvariable=self.seq_var)
        seq_spin.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        
        # Neurons
        ttk.Label(model_frame, text="Neurons:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.neurons_var = tk.IntVar(value=50)
        neurons_spin = ttk.Spinbox(model_frame, from_=10, to=200, textvariable=self.neurons_var)
        neurons_spin.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        
        # Layers
        ttk.Label(model_frame, text="Layers:").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.layers_var = tk.IntVar(value=2)
        layers_spin = ttk.Spinbox(model_frame, from_=1, to=5, textvariable=self.layers_var)
        layers_spin.grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        
        # Dropout
        ttk.Label(model_frame, text="Dropout:").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        self.dropout_var = tk.DoubleVar(value=0.2)
        dropout_spin = ttk.Spinbox(model_frame, from_=0.0, to=0.5, increment=0.1, 
                                   textvariable=self.dropout_var)
        dropout_spin.grid(row=4, column=1, sticky="ew", padx=5, pady=2)
        
        # Training parameters
        train_frame = ttk.Frame(self)
        train_frame.pack(fill="x", padx=5, pady=5)
        
        # Epochs
        ttk.Label(train_frame, text="Epochs:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.epochs_var = tk.IntVar(value=50)
        epochs_spin = ttk.Spinbox(train_frame, from_=10, to=500, textvariable=self.epochs_var)
        epochs_spin.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        
        # Batch size
        ttk.Label(train_frame, text="Batch Size:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.batch_var = tk.IntVar(value=32)
        batch_spin = ttk.Spinbox(train_frame, from_=8, to=128, textvariable=self.batch_var)
        batch_spin.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        
        # Learning rate
        ttk.Label(train_frame, text="Learning Rate:").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.lr_var = tk.DoubleVar(value=0.001)
        lr_spin = ttk.Spinbox(train_frame, from_=0.0001, to=0.01, increment=0.0001, 
                              textvariable=self.lr_var)
        lr_spin.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        
        # Training Results section
        results_frame = ttk.LabelFrame(self, text="Training Results")
        results_frame.pack(fill="x", padx=5, pady=5)
        
        # Create a frame for the listbox and scrollbar
        model_frame = ttk.Frame(results_frame)
        model_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create scrollbar
        scrollbar = ttk.Scrollbar(model_frame)
        scrollbar.pack(side="right", fill="y")
        
        # Create listbox
        self.model_listbox = tk.Listbox(model_frame, 
                                       yscrollcommand=scrollbar.set,
                                       height=4)
        self.model_listbox.pack(side="left", fill="both", expand=True)
        
        # Configure scrollbar
        scrollbar.config(command=self.model_listbox.yview)
        
        # Button frame
        button_frame = ttk.Frame(results_frame)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        # View button
        self.view_model_btn = ttk.Button(button_frame, text="View", command=self._on_view_model_clicked)
        self.view_model_btn.pack(side="left", fill="x", expand=True, padx=2, pady=2)
        
        # Refresh button
        refresh_models_btn = ttk.Button(button_frame, text="Refresh", command=self.refresh_models)
        refresh_models_btn.pack(side="right", fill="x", expand=True, padx=2, pady=2)
        
        # Action buttons frame
        action_frame = ttk.Frame(self)
        action_frame.pack(fill="x", padx=5, pady=5)
        
        # Train button
        self.train_btn = ttk.Button(action_frame, text="Train Model", command=self._on_train_clicked)
        self.train_btn.pack(fill="x", pady=2)
        
        # Save model button
        self.save_model_btn = ttk.Button(action_frame, text="Save Model", command=self._on_save_model_clicked)
        self.save_model_btn.pack(fill="x", pady=2)
        
        # Load model button
        self.load_model_btn = ttk.Button(action_frame, text="Load Model", command=self._on_load_model_clicked)
        self.load_model_btn.pack(fill="x", pady=2)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(self, textvariable=self.status_var)
        status_label.pack(fill="x", padx=5, pady=5)
        
        # Initial refresh of model list
        self.refresh_models()
        
    def _on_train_clicked(self):
        """Handle train button click"""
        # Get database controls to get selected values
        database_controls = self.parent.winfo_children()[0]
        if isinstance(database_controls, ttk.Frame):
            database_controls = database_controls.winfo_children()[0]
        
        db_name = database_controls.db_var.get()
        table_name = database_controls.table_var.get()
        tickers = database_controls.get_selected_tickers()
        
        # Validate parameters
        if not db_name:
            self.set_status("Please select a database")
            return
            
        if not table_name:
            self.set_status("Please select a table")
            return
            
        if not tickers:
            self.set_status("Please select at least one ticker")
            return
            
        # Get training parameters
        params = {
            'db_name': db_name,
            'table_name': table_name,
            'tickers': tickers,
            'model_type': self.model_var.get(),
            'sequence_length': self.seq_var.get(),
            'neurons': self.neurons_var.get(),
            'layers': self.layers_var.get(),
            'dropout': self.dropout_var.get(),
            'epochs': self.epochs_var.get(),
            'batch_size': self.batch_var.get(),
            'learning_rate': self.lr_var.get()
        }
        
        # Publish event to train model
        self.event_bus.publish("train_model", params)
        
        # Update status
        self.set_status("Training started...")
    
    def _on_save_model_clicked(self):
        """Handle save model button click"""
        print("Save Model button clicked")
        
        # Get selected model from listbox
        selected_indices = self.model_listbox.curselection()
        if not selected_indices:
            messagebox.showinfo("Information", "Please select a model from the list")
            return
            
        selected_model = self.model_listbox.get(selected_indices[0])
        print(f"Selected model: {selected_model}")
        
        # Show acknowledgment that model is being saved
        self.set_status("Saving model to 'models' directory...")
        print("Saving model to 'models' directory...")
        messagebox.showinfo("Saving Model", "Model will be saved to the 'models' directory")
        
        # Publish event to save model
        self.event_bus.publish("save_model", {'model_name': selected_model})
    
    def _on_load_model_clicked(self):
        """Handle load model button click"""
        print("Load Model button clicked")
        
        # Ask user to select a model file
        model_file = filedialog.askopenfilename(
            title="Select a model file",
            filetypes=[("Keras models", "*.h5 *.keras"), ("All files", "*.*")]
        )
        
        if not model_file:
            return
            
        print(f"Loading model from {model_file}")
        self.set_status(f"Loading model from {model_file}...")
        
        # Publish event to load model
        self.event_bus.publish("load_model", {'model_path': model_file})
    
    def _on_view_model_clicked(self):
        """Handle view model button click"""
        print("View Model button clicked")
        selected_indices = self.model_listbox.curselection()
        
        if not selected_indices:
            messagebox.showinfo("Information", "Please select a model from the list")
            return
        
        selected_model = self.model_listbox.get(selected_indices[0])
        print(f"Selected model: {selected_model}")
        
        # Publish event to view model
        self.event_bus.publish("view_model", {'model_name': selected_model})
        self.set_status(f"Viewing model: {selected_model}")
    
    def refresh_models(self):
        """Refresh the list of trained models"""
        print("Refreshing model list")
        self.model_listbox.delete(0, tk.END)
        
        # This is a placeholder - you'll need to implement the actual model discovery
        import os
        models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5') or f.endswith('.keras')]
            for model_file in model_files:
                self.model_listbox.insert(tk.END, model_file)
        
        if self.model_listbox.size() == 0:
            self.model_listbox.insert(tk.END, "No models found")
            self.view_model_btn.config(state="disabled")
            self.save_model_btn.config(state="disabled")
        else:
            self.view_model_btn.config(state="normal")
            self.save_model_btn.config(state="normal")
        
        self.set_status("Model list refreshed")
    
    def _on_model_saved(self, data):
        """Handle model saved event"""
        model_path = data.get('model_path', 'unknown')
        print(f"Model saved successfully to {model_path}")
        messagebox.showinfo("Success", f"Model saved successfully to {model_path}")
        self.set_status(f"Model saved successfully to {model_path}")
        self.refresh_models()
    
    def _on_model_loaded(self, data):
        """Handle model loaded event"""
        model_path = data.get('model_path', 'unknown')
        print(f"Model loaded successfully from {model_path}")
        messagebox.showinfo("Success", f"Model loaded successfully from {model_path}")
        self.set_status(f"Model loaded successfully from {model_path}")
    
    def _on_model_save_error(self, data):
        """Handle model save error event"""
        error_msg = data.get('message', 'Unknown error')
        print(f"Error saving model: {error_msg}")
        messagebox.showerror("Error", f"Failed to save model: {error_msg}")
        self.set_status(f"Error saving model: {error_msg}")
    
    def _on_model_load_error(self, data):
        """Handle model load error event"""
        error_msg = data.get('message', 'Unknown error')
        print(f"Error loading model: {error_msg}")
        messagebox.showerror("Error", f"Failed to load model: {error_msg}")
        self.set_status(f"Error loading model: {error_msg}")
        
    def set_status(self, message):
        """Set status message"""
        self.status_var.set(message) 