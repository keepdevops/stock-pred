"""
Main application window
"""
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import traceback
import os

from config.settings import WINDOW_WIDTH, WINDOW_HEIGHT, DARK_BG
from data.database import find_databases
from ui.styles import configure_styles
from .control_panel import ControlPanel
from .visualization_panel import VisualizationPanel

class StockAnalyzerApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Stock Market Analyzer")
        self.root.geometry(f'{WINDOW_WIDTH}x{WINDOW_HEIGHT}')
        self.root.configure(bg=DARK_BG)
        
        # Configure styles
        self.style = configure_styles(self.root)
        
        # Determine the data directory
        current_dir = os.getcwd()
        if os.path.basename(current_dir) == 'stock-analyzer':
            self.data_dir = os.path.join(os.path.dirname(current_dir), 'data')
        else:
            self.data_dir = os.path.join(current_dir, 'data')
        
        # Find available databases
        self.databases = find_databases(self.data_dir)
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create control panel
        self.control_panel = ControlPanel(self.main_frame, self.databases, self.data_dir)
        
        # Create visualization panel
        self.visualization_panel = VisualizationPanel(self.main_frame)
        
        # Set up event bindings
        self.control_panel.set_train_callback(self.on_train)
        self.control_panel.set_predict_callback(self.on_predict)
        
        # Global variables for trained model and scaler
        self.trained_model = None
        self.trained_scaler = None
        self.sequence_length = 10
    
    def on_train(self, params):
        """Handle training request"""
        # Implementation will be added later
        pass
        
    def on_predict(self, params):
        """Handle prediction request"""
        # Implementation will be added later
        pass
        
    def run(self):
        """Run the application"""
        self.root.mainloop()
