import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
from datetime import datetime, timedelta
import re
import duckdb

# Establish a connection to your DuckDB database
conn = duckdb.connect('your_database.duckdb')  # Replace with your actual database file

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
    # Construct the SQL query to select the specified columns
    query = f"SELECT {', '.join(columns)} FROM {table_name}"
    print(f"Executing query: {query}")  # Debug: Print the query being executed
    
    # Execute the query and fetch the results into a DataFrame
    df = connection.execute(query).fetchdf()
    print(df.head())  # Debug: Print the first few rows to verify the data
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
    def __init__(self, root, tickers, table, connection, fields):
        self.root = root
        self.tickers = tickers
        self.table = table
        self.connection = connection
        self.fields = fields
        self.ticker_column = self.detect_ticker_column()  # Set ticker_column here

    def create_plot(self):
        # Fetch data and plot
        data = self.fetch_data()
        for ticker, df in data.items():
            self.plot_data(df, ticker)

    def fetch_data(self):
        # Fetch data from the database
        data = {}
        for ticker in self.tickers:
            query = f"SELECT {', '.join(self.fields)} FROM {self.table} WHERE {self.ticker_column} = ?"
            df = self.connection.execute(query, [ticker]).df()
            data[ticker] = df
        return data

    def plot_data(self, df, ticker):
        plt.figure(figsize=(10, 6))
        for field in self.fields:
            if field != 'date':
                plt.plot(df['date'], df[field], label=field)
        plt.title(f"Data for {ticker}")
        plt.xlabel('Date')
        plt.ylabel('Values')
        plt.legend()
        plt.show()

    def detect_ticker_column(self):
        """Detect the ticker column in the selected table"""
        try:
            # Get column names
            columns = self.connection.execute(f"PRAGMA table_info({self.table})").df()['name'].values
            print(f"Available columns: {columns}")
            
            # Look for common ticker column names
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
        """Update fields and refresh UI when table or sector changes"""
        if self.table != new_table:
            print(f"Switching from {self.table} to {new_table}")
            self.table = new_table
            self.ticker_column = self.detect_ticker_column()
            self.create_metrics_selector()

    def create_metrics_selector(self):
        """Create metrics selection frame"""
        print("Creating metrics selector for table:", self.table)
        
        # Clear existing metrics frame if it exists
        if hasattr(self, 'metrics_frame'):
            self.metrics_frame.destroy()
        
        # Create a new frame for metrics selection
        self.metrics_frame = ttk.Frame(self.main_frame)
        self.metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Fetch available fields for the selected table
        available_fields = self.fields
        
        # Create a dictionary to hold the state of each metric checkbox
        self.metric_vars = {}
        
        for field in available_fields:
            var = tk.BooleanVar(value=field in self.fields)  # Default to selected if in fields
            chk = ttk.Checkbutton(self.metrics_frame, text=field.replace('_', ' ').title(), variable=var)
            chk.pack(side=tk.LEFT, padx=5, pady=5)
            self.metric_vars[field] = var
        
        print("Metrics selector created for table:", self.table)