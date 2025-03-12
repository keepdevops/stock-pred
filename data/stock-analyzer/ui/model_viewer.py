"""
Model Viewer - UI component for viewing saved models with .keras extension
"""
import os
import glob
import pickle
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf

class ModelViewer(ttk.Frame):
    """Model Viewer component that can be embedded in any tkinter window"""
    
    def __init__(self, parent, models_dir="/Users/moose/stock-pred/data/stock-analyzer/models"):
        super().__init__(parent)
        self.models_dir = models_dir
        
        # Create control frame
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Ticker selection
        ttk.Label(control_frame, text="Ticker:").pack(side=tk.LEFT, padx=5)
        self.ticker_var = tk.StringVar()
        self.ticker_combo = ttk.Combobox(control_frame, textvariable=self.ticker_var, width=10)
        self.ticker_combo.pack(side=tk.LEFT, padx=5)
        
        # Model type selection
        ttk.Label(control_frame, text="Model Type:").pack(side=tk.LEFT, padx=5)
        self.model_type_var = tk.StringVar(value="lstm")
        model_type_combo = ttk.Combobox(control_frame, textvariable=self.model_type_var, 
                                        values=["lstm", "gru"], width=5)
        model_type_combo.pack(side=tk.LEFT, padx=5)
        
        # View button
        view_button = ttk.Button(control_frame, text="View Model", command=self.view_selected_model)
        view_button.pack(side=tk.LEFT, padx=10)
        
        # Refresh button
        refresh_button = ttk.Button(control_frame, text="Refresh List", command=self.refresh_models)
        refresh_button.pack(side=tk.LEFT, padx=5)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Summary tab
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        
        # Create summary text widget
        self.summary_text = tk.Text(self.summary_frame, wrap=tk.WORD, width=80, height=20)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # History tab
        self.history_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.history_frame, text="Training History")
        
        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Initialize
        self.refresh_models()
    
    def refresh_models(self):
        """Refresh the list of available models"""
        self.status_var.set("Refreshing model list...")
        
        # Find all models with _model.keras pattern
        pattern = os.path.join(self.models_dir, "*_model.keras")
        model_files = glob.glob(pattern)
        
        # Extract tickers from filenames
        tickers = set()
        for model_file in model_files:
            basename = os.path.basename(model_file)
            parts = basename.split('_')
            if len(parts) >= 3:
                tickers.add(parts[0])
        
        # Update ticker combobox
        self.ticker_combo['values'] = sorted(list(tickers))
        if tickers:
            self.ticker_var.set(next(iter(tickers)))
        
        self.status_var.set(f"Found {len(model_files)} models for {len(tickers)} tickers")
    
    def view_selected_model(self):
        """View the selected model"""
        ticker = self.ticker_var.get()
        model_type = self.model_type_var.get()
        
        if not ticker:
            self.status_var.set("Please select a ticker")
            return
        
        self.status_var.set(f"Loading model for {ticker}...")
        
        # Clear existing content
        self.summary_text.delete(1.0, tk.END)
        for widget in self.history_frame.winfo_children():
            widget.destroy()
        
        # Find model file
        model_filename = f"{ticker}_{model_type}_model.keras"
        model_path = os.path.join(self.models_dir, model_filename)
        history_path = model_path.replace(".keras", "_history.pkl")
        
        # Check if model exists
        if not os.path.exists(model_path):
            self.status_var.set(f"Model not found: {model_filename}")
            
            # Try finding any model with this ticker
            pattern = os.path.join(self.models_dir, f"{ticker}_*.keras")
            matches = glob.glob(pattern)
            if matches:
                model_path = matches[0]
                history_path = model_path.replace(".keras", "_history.pkl")
                self.status_var.set(f"Found alternative model: {os.path.basename(model_path)}")
            else:
                self.status_var.set(f"No models found for ticker: {ticker}")
                return
        
        # Load the model
        try:
            model = tf.keras.models.load_model(model_path)
            
            # Display model summary
            self.summary_text.insert(tk.END, f"Model: {os.path.basename(model_path)}\n\n")
            
            # Capture model summary
            stringlist = []
            model.summary(print_fn=lambda x: stringlist.append(x))
            model_summary = "\n".join(stringlist)
            self.summary_text.insert(tk.END, model_summary)
            
            # Add file info
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            self.summary_text.insert(tk.END, f"\n\nModel File Size: {model_size:.2f} MB\n")
            self.summary_text.insert(tk.END, f"Model Path: {model_path}\n")
            
        except Exception as e:
            self.status_var.set(f"Error loading model: {e}")
            self.summary_text.insert(tk.END, f"Error loading model: {str(e)}")
            return
        
        # Load and display training history
        if os.path.exists(history_path):
            try:
                with open(history_path, "rb") as f:
                    history = pickle.load(f)
                
                # Create matplotlib figure
                fig = plt.Figure(figsize=(8, 6), dpi=100)
                
                # Plot loss
                ax1 = fig.add_subplot(111)
                ax1.plot(history['loss'], label='Training Loss', color='blue')
                if 'val_loss' in history:
                    ax1.plot(history['val_loss'], label='Validation Loss', color='red')
                ax1.set_title(f"{ticker} Model Loss")
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()
                ax1.grid(True, linestyle='--', alpha=0.7)
                
                # Create Matplotlib canvas
                canvas = FigureCanvasTkAgg(fig, self.history_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
                
                self.status_var.set(f"Model loaded successfully: {os.path.basename(model_path)}")
                
            except Exception as e:
                self.status_var.set(f"Error loading history: {e}")
        else:
            self.status_var.set(f"No training history found for {ticker}")

# Function to open model viewer window (separate from the class)
def open_model_viewer_window(parent):
    """Open a standalone model viewer window"""
    viewer_window = tk.Toplevel(parent)
    viewer_window.title("Model Viewer")
    viewer_window.geometry("800x600")
    
    # Create and pack the model viewer
    viewer = ModelViewer(viewer_window)
    viewer.pack(fill=tk.BOTH, expand=True)
    
    return viewer_window