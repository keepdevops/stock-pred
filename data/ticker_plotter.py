import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
from datetime import datetime, timedelta
import re
import duckdb
import os
from matplotlib.figure import Figure

def plot_data(df, x_column, y_column):
    if x_column not in df.columns:
        raise ValueError(f"Column '{x_column}' not found in DataFrame")
    
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_column], df[y_column], marker='o')
    plt.title(f"{y_column} over {x_column}")
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.grid(True)
    plt.show()

def fetch_data(connection, table_name, columns):
    query = f"SELECT {', '.join(columns)} FROM {table_name}"
    print(f"Executing query: {query}")
    df = connection.execute(query).fetchdf()
    print(df.head())
    return df

table_fields = {
    'balance_sheets': {
        'ticker_column': 'symbol',
        'fields': ['symbol', 'sector', 'date', 'period', 'fiscal_quarter', 'fiscal_year',
                   'total_assets', 'total_liabilities', 'total_equity', 'cash_equivalents',
                   'total_debt', 'working_capital', 'updated_at']
    },
    'historical_commodities': {
        'ticker_column': 'symbol',
        'fields': ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'sector', 'updated_at']
    },
    'historical_forex': {
        'ticker_column': 'pair',
        'fields': ['pair', 'date', 'open', 'high', 'low', 'close', 'volume', 'category', 'updated_at', 'sector']
    },
    # Add other tables as needed, ensuring 'date' is included where applicable
}

class TickerPlotter:
    def __init__(self, parent):
        self.parent = parent
        
        # Create figure
        self.fig = Figure(figsize=(10, 6))
        self.ax = self.fig.add_subplot(111)
        
        # Create toolbar frame
        self.toolbar_frame = ttk.Frame(parent)
        self.toolbar_frame.grid(row=0, column=0, sticky="ew")
        
        # Create canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        
        # Create toolbar with the toolbar frame as parent
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.grid(row=0, column=0, sticky="ew")
        
        # Configure grid weights
        parent.grid_rowconfigure(1, weight=1)  # Canvas expands vertically
        parent.grid_columnconfigure(0, weight=1)  # Canvas expands horizontally
        self.toolbar_frame.grid_columnconfigure(0, weight=1)  # Toolbar expands horizontally
        
    def connect_to_db(self, db_path):
        """Connect to the specified DuckDB database."""
        if not os.path.exists(db_path):
            messagebox.showinfo("Database Selection", "Database file does not exist. Please select an existing database.")
            db_path = filedialog.askopenfilename(
                title="Select DuckDB Database",
                filetypes=[("DuckDB files", "*.db"), ("All files", "*.*")]
            )
            if not db_path:
                raise FileNotFoundError("No database file selected.")
        
        try:
            print(f"Connecting to DuckDB database at: {db_path}")
            connection = duckdb.connect(db_path)
            print("Connection successful.")
            return connection
        except Exception as e:
            print(f"Error connecting to database: {e}")
            messagebox.showerror("Database Error", f"Failed to connect to database: {e}")
            raise

    def create_plot(self):
        data = self.fetch_data()
        for ticker, df in data.items():
            self.plot_data(df, ticker)

    def fetch_data(self):
        data = {}
        for ticker in self.tickers:
            query = f"SELECT {', '.join(self.fields)} FROM {self.table} WHERE {self.ticker_column} = ?"
            df = self.connection.execute(query, [ticker]).df()
            data[ticker] = df
        return data

    def plot_data(self, df, ticker):
        """Plot the selected data"""
        self.ax.clear()
        
        # Convert date column to datetime if it's not already
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            
        # Plot each selected field
        for field in self.fields:
            self.ax.plot(df['date'], df[field], label=field)
            
        # Customize the plot
        self.ax.set_title(f'{ticker} Data')
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Value')
        self.ax.legend()
        self.ax.grid(True)
        
        # Rotate x-axis labels for better readability
        self.ax.tick_params(axis='x', rotation=45)
        
        # Adjust layout to prevent label cutoff
        self.fig.tight_layout()
        
        # Redraw the canvas
        self.canvas.draw()

    def detect_ticker_column(self):
        try:
            columns = self.connection.execute(f"PRAGMA table_info({self.table})").df()['name'].values
            print(f"Available columns: {columns}")
            
            if self.table == 'historical_forex' and 'pair' in columns:
                print("Using 'pair' column for historical_forex table.")
                return 'pair'
            
            for col in ['symbol', 'ticker']:
                if col in columns:
                    print(f"Using '{col}' as ticker identifier")
                    return col
            
            raise ValueError("No standard ticker column found")
            
        except Exception as e:
            print(f"Error detecting ticker column: {e}")
            raise

    def update_fields_and_refresh_ui(self, new_table):
        if self.table != new_table:
            print(f"Switching from {self.table} to {new_table}")
            self.table = new_table
            self.ticker_column = self.detect_ticker_column()
            self.create_metrics_selector()

    def create_metrics_selector(self):
        print("Creating metrics selector for table:", self.table)
        
        if hasattr(self, 'metrics_frame'):
            self.metrics_frame.destroy()
        
        self.metrics_frame = ttk.Frame(self.main_frame)
        self.metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        available_fields = self.fields
        self.metric_vars = {}
        
        for field in available_fields:
            var = tk.BooleanVar(value=field in self.fields)
            chk = ttk.Checkbutton(self.metrics_frame, text=field.replace('_', ' ').title(), variable=var)
            chk.pack(side=tk.LEFT, padx=5, pady=5)
            self.metric_vars[field] = var
        
        print("Metrics selector created for table:", self.table)