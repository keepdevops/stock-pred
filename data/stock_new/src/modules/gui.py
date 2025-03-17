import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Optional

from ..modules.database import DatabaseConnector
from ..modules.data_adapter import DataAdapter
from ..modules.stock_ai_agent import StockAIAgent

class StockGUI:
    def __init__(
        self,
        root: tk.Tk,
        db_connector: DatabaseConnector,
        data_adapter: DataAdapter,
        ai_agent: StockAIAgent
    ):
        self.root = root
        self.db_connector = db_connector
        self.data_adapter = data_adapter
        self.ai_agent = ai_agent
        
        # Initialize GUI components
        self.setup_left_panel()
        self.setup_right_panel()
    
    def setup_left_panel(self):
        """Setup left panel with controls"""
        left_frame = ttk.Frame(self.root)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Database selection
        ttk.Label(left_frame, text="Database:").pack(anchor=tk.W)
        self.db_combo = ttk.Combobox(left_frame, state="readonly")
        self.db_combo.pack(fill=tk.X, pady=2)
        self.db_combo.bind("<<ComboboxSelected>>", self.on_database_selected)
        
        # Table selection
        ttk.Label(left_frame, text="Sector Table:").pack(anchor=tk.W)
        self.table_combo = ttk.Combobox(left_frame, state="readonly")
        self.table_combo.pack(fill=tk.X, pady=2)
        self.table_combo.bind("<<ComboboxSelected>>", self.on_table_selected)
        
        # Add more controls here...
    
    def setup_right_panel(self):
        """Setup right panel with plots"""
        right_frame = ttk.Frame(self.root)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Setup matplotlib figure
        self.figure = plt.Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, right_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def on_database_selected(self, event):
        """Handle database selection"""
        selected_db = self.db_combo.get()
        if selected_db:
            self.db_connector.create_connection(selected_db)
            tables = self.db_connector.get_tables()
            self.table_combo["values"] = tables
    
    def on_table_selected(self, event):
        """Handle table selection"""
        selected_table = self.table_combo.get()
        if selected_table:
            # Load and display data
            pass 