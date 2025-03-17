import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
from app.config.template_manager import TemplateManager

class StockGUI:
    def __init__(self, root):
        self.root = root
        self.template_manager = TemplateManager()
        
        # Create main frames
        self.create_frames()
        # Create basic controls
        self.create_controls()
        # Create plot area
        self.create_plot()
        # Create template controls
        self.create_template_controls()
    
    def create_frames(self):
        # Left panel for controls
        self.left_frame = ttk.Frame(self.root, padding="5")
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        # Right panel for plots
        self.right_frame = ttk.Frame(self.root, padding="5")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    
    def create_controls(self):
        # Database selection
        ttk.Label(self.left_frame, text="Database:").pack(anchor=tk.W)
        self.db_combo = ttk.Combobox(self.left_frame, values=["Sample DB 1", "Sample DB 2"])
        self.db_combo.pack(fill=tk.X, pady=2)
        
        # Table selection
        ttk.Label(self.left_frame, text="Sector:").pack(anchor=tk.W)
        self.table_combo = ttk.Combobox(self.left_frame, values=["Tech", "Crypto", "Finance"])
        self.table_combo.pack(fill=tk.X, pady=2)
        
        # Ticker selection
        ttk.Label(self.left_frame, text="Ticker:").pack(anchor=tk.W)
        self.ticker_combo = ttk.Combobox(self.left_frame, values=["AAPL", "GOOGL", "MSFT"])
        self.ticker_combo.pack(fill=tk.X, pady=2)
        
        # Model selection
        ttk.Label(self.left_frame, text="Model:").pack(anchor=tk.W)
        self.model_combo = ttk.Combobox(
            self.left_frame,
            values=["LSTM", "GRU", "BiLSTM", "CNN-LSTM", "Transformer"]
        )
        self.model_combo.pack(fill=tk.X, pady=2)
        
        # Buttons
        ttk.Button(self.left_frame, text="Train Model", command=self.on_train).pack(fill=tk.X, pady=2)
        ttk.Button(self.left_frame, text="Predict", command=self.on_predict).pack(fill=tk.X, pady=2)
        
        # Status area
        self.status_text = tk.Text(self.left_frame, height=5, width=30)
        self.status_text.pack(fill=tk.X, pady=2)
    
    def create_plot(self):
        # Create matplotlib figure
        self.figure = plt.Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, self.right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add sample plot
        ax = self.figure.add_subplot(111)
        ax.set_title("Stock Price Analysis")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        self.canvas.draw()
    
    def create_template_controls(self):
        """Create template management controls"""
        # Template frame
        template_frame = ttk.LabelFrame(self.left_frame, text="Templates", padding="5")
        template_frame.pack(fill=tk.X, pady=5)
        
        # Template selection
        ttk.Label(template_frame, text="Load Template:").pack(anchor=tk.W)
        self.template_combo = ttk.Combobox(
            template_frame,
            values=self.get_template_names()
        )
        self.template_combo.pack(fill=tk.X, pady=2)
        self.template_combo.bind('<<ComboboxSelected>>', self.on_template_selected)
        
        # Save template
        save_frame = ttk.Frame(template_frame)
        save_frame.pack(fill=tk.X, pady=2)
        
        self.template_name = ttk.Entry(save_frame)
        self.template_name.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        ttk.Button(
            save_frame,
            text="Save",
            command=self.on_save_template
        ).pack(side=tk.RIGHT, padx=2)
    
    def get_template_names(self):
        """Get list of available templates"""
        template_dir = Path(self.template_manager.templates_dir)
        return [f.stem for f in template_dir.glob("*.json")]
    
    def on_template_selected(self, event):
        """Handle template selection"""
        template_name = self.template_combo.get()
        if template_name:
            config = self.template_manager.load_template(template_name)
            if config:
                self.apply_template_config(config)
                self.status_text.insert(tk.END, f"Loaded template: {template_name}\n")
                self.status_text.see(tk.END)
    
    def on_save_template(self):
        """Handle template saving"""
        name = self.template_name.get()
        if not name:
            messagebox.showerror("Error", "Please enter a template name")
            return
        
        # Gather current configuration
        config = self.get_current_config()
        
        # Save template
        if self.template_manager.save_template(name, config):
            messagebox.showinfo("Success", f"Template '{name}' saved successfully")
            # Update template list
            self.template_combo['values'] = self.get_template_names()
        else:
            messagebox.showerror("Error", f"Failed to save template '{name}'")
    
    def get_current_config(self):
        """Gather current GUI configuration"""
        return {
            "models": {
                self.model_combo.get(): {
                    "training": {
                        "epochs": 100,  # Add proper input fields for these
                        "batch_size": 32,
                        "validation_split": 0.2
                    }
                }
            },
            "data_processing": {
                "sequence_length": 10,
                "features": ["close", "volume"]  # Add proper selection for these
            }
        }
    
    def apply_template_config(self, config):
        """Apply template configuration to GUI"""
        # Set model selection if available
        if "models" in config:
            model_names = list(config["models"].keys())
            if model_names:
                self.model_combo['values'] = model_names
                self.model_combo.set(model_names[0])
        
        # Update other GUI elements based on config
        if "data_processing" in config:
            self.status_text.insert(tk.END, "Applied data processing config\n")
        
        # Update visualization settings if available
        if "visualization" in config:
            self.update_plot_settings(config["visualization"])
    
    def update_plot_settings(self, viz_config):
        """Update plot appearance based on configuration"""
        colors = viz_config.get("colors", {})
        if colors:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.set_title("Stock Price Analysis")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            # Apply colors and other settings
            self.canvas.draw()
    
    def on_train(self):
        """Handle train button click"""
        self.status_text.insert(tk.END, "Training model...\n")
        self.status_text.see(tk.END)
    
    def on_predict(self):
        """Handle predict button click"""
        self.status_text.insert(tk.END, "Making predictions...\n")
        self.status_text.see(tk.END) 