import tkinter as tk
from tkinter import ttk, messagebox
import logging
from datetime import datetime, timedelta
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..modules.data_loader import DataLoader
from ..modules.database import Database

class MainWindow:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Stock Market Analyzer")
        self.root.geometry("1200x800")
        
        self.logger = logging.getLogger(__name__)
        self.data_loader = DataLoader()
        self.database = Database()
        
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the user interface."""
        # Create main container
        self.main_container = ttk.Frame(self.root, padding="10")
        self.main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_container.columnconfigure(1, weight=1)
        self.main_container.rowconfigure(1, weight=1)
        
        # Create left panel
        self.create_left_panel()
        
        # Create right panel
        self.create_right_panel()
        
    def create_left_panel(self):
        """Create the left control panel."""
        left_panel = ttk.Frame(self.main_container)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Stock symbol entry
        ttk.Label(left_panel, text="Stock Symbol:").grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.symbol_var = tk.StringVar()
        self.symbol_entry = ttk.Entry(left_panel, textvariable=self.symbol_var)
        self.symbol_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Date range selection
        ttk.Label(left_panel, text="Date Range:").grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.date_range_var = tk.StringVar(value="1Y")
        date_ranges = ["1M", "3M", "6M", "1Y", "5Y", "MAX"]
        self.date_range_combo = ttk.Combobox(left_panel, textvariable=self.date_range_var, values=date_ranges)
        self.date_range_combo.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(0, 5))
        
        # Fetch button
        self.fetch_button = ttk.Button(left_panel, text="Fetch Data", command=self.fetch_data)
        self.fetch_button.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Company info frame
        company_frame = ttk.LabelFrame(left_panel, text="Company Information", padding="5")
        company_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.company_info_text = tk.Text(company_frame, height=10, width=30)
        self.company_info_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
    def create_right_panel(self):
        """Create the right panel with charts and data."""
        right_panel = ttk.Frame(self.main_container)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)
        
        # Create tabs
        self.create_chart_tab()
        self.create_data_tab()
        
    def create_chart_tab(self):
        """Create the chart tab."""
        chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(chart_frame, text="Charts")
        
        # Configure grid weights
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.figure, master=chart_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
    def create_data_tab(self):
        """Create the data tab."""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="Data")
        
        # Configure grid weights
        data_frame.columnconfigure(0, weight=1)
        data_frame.rowconfigure(0, weight=1)
        
        # Create treeview
        self.tree = ttk.Treeview(data_frame, columns=("Date", "Open", "High", "Low", "Close", "Volume"))
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure columns
        self.tree.heading("Date", text="Date")
        self.tree.heading("Open", text="Open")
        self.tree.heading("High", text="High")
        self.tree.heading("Low", text="Low")
        self.tree.heading("Close", text="Close")
        self.tree.heading("Volume", text="Volume")
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(data_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.tree.configure(yscrollcommand=scrollbar.set)
        
    def fetch_data(self):
        """Fetch stock data and update the UI."""
        symbol = self.symbol_var.get().strip().upper()
        if not symbol:
            messagebox.showerror("Error", "Please enter a stock symbol")
            return
            
        # Validate symbol
        is_valid, error_msg = self.data_loader.validate_symbol(symbol)
        if not is_valid:
            messagebox.showerror("Error", error_msg)
            return
            
        # Get date range
        date_range = self.date_range_var.get()
        end_date = datetime.now()
        if date_range == "1M":
            start_date = end_date - timedelta(days=30)
        elif date_range == "3M":
            start_date = end_date - timedelta(days=90)
        elif date_range == "6M":
            start_date = end_date - timedelta(days=180)
        elif date_range == "1Y":
            start_date = end_date - timedelta(days=365)
        elif date_range == "5Y":
            start_date = end_date - timedelta(days=1825)
        else:  # MAX
            start_date = None
            
        # Fetch data
        df, error_msg = self.data_loader.fetch_stock_data(symbol, start_date, end_date)
        if df is None:
            messagebox.showerror("Error", error_msg)
            return
            
        # Update company info
        company_info, error_msg = self.data_loader.get_company_info(symbol)
        if company_info:
            self.update_company_info(company_info)
            
        # Update chart
        self.update_chart(df)
        
        # Update data table
        self.update_data_table(df)
        
        # Save to database
        self.database.save_stock_data(df, symbol)
        
    def update_company_info(self, info: dict):
        """Update the company information display."""
        self.company_info_text.delete(1.0, tk.END)
        self.company_info_text.insert(tk.END, f"Name: {info['name']}\n")
        self.company_info_text.insert(tk.END, f"Sector: {info['sector']}\n")
        self.company_info_text.insert(tk.END, f"Industry: {info['industry']}\n")
        self.company_info_text.insert(tk.END, f"Market Cap: ${info['market_cap']:,.2f}\n")
        self.company_info_text.insert(tk.END, f"P/E Ratio: {info['pe_ratio']:.2f}\n")
        self.company_info_text.insert(tk.END, f"Dividend Yield: {info['dividend_yield']:.2%}\n")
        self.company_info_text.insert(tk.END, f"Beta: {info['beta']:.2f}\n")
        self.company_info_text.insert(tk.END, f"52W High: ${info['fifty_two_week_high']:.2f}\n")
        self.company_info_text.insert(tk.END, f"52W Low: ${info['fifty_two_week_low']:.2f}\n")
        
    def update_chart(self, df: pd.DataFrame):
        """Update the stock price chart."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(df.index, df['close'], label='Close Price')
        ax.set_title('Stock Price History')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.grid(True)
        ax.legend()
        self.canvas.draw()
        
    def update_data_table(self, df: pd.DataFrame):
        """Update the data table with stock data."""
        # Clear existing items
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        # Insert new data
        for index, row in df.iterrows():
            self.tree.insert("", tk.END, values=(
                index.strftime('%Y-%m-%d'),
                f"${row['open']:.2f}",
                f"${row['high']:.2f}",
                f"${row['low']:.2f}",
                f"${row['close']:.2f}",
                f"{row['volume']:,}"
            ))
            
    def run(self):
        """Start the application."""
        self.root.mainloop()
        
    def cleanup(self):
        """Clean up resources."""
        self.database.close() 