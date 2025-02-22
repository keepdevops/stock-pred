import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import duckdb
import pandas as pd
from datetime import datetime, timedelta

class TickerPlotter:
    def __init__(self, root, selected_tickers, selected_table):
        self.root = root
        self.selected_tickers = selected_tickers
        self.selected_table = selected_table
        
        # Create a new top-level window
        self.plot_window = tk.Toplevel(root)
        self.plot_window.title("Ticker Plotter")
        self.plot_window.geometry("800x600")
        
        # Database connection
        try:
            self.conn = duckdb.connect('historical_market_data.db', read_only=True)
            print("Successfully connected to database")
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to connect to database: {e}")
            raise

        # Create main frame
        self.main_frame = ttk.Frame(self.plot_window, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.plot_window.grid_rowconfigure(0, weight=1)
        self.plot_window.grid_columnconfigure(0, weight=1)
        
        # Setup widgets and create initial plot
        self.setup_widgets()
        self.create_plot()

    def setup_widgets(self):
        # Control Frame
        control_frame = ttk.Frame(self.main_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        # Time range selection
        ttk.Label(control_frame, text="Time Range:").pack(side=tk.LEFT, padx=5)
        self.time_range = tk.StringVar(value="1Y")
        time_choices = ["1M", "3M", "6M", "1Y", "3Y", "5Y", "ALL"]
        time_menu = ttk.Combobox(control_frame, textvariable=self.time_range, values=time_choices, width=5)
        time_menu.pack(side=tk.LEFT, padx=5)
        
        # Column selection (for numerical columns)
        ttk.Label(control_frame, text="Column:").pack(side=tk.LEFT, padx=5)
        self.column_var = tk.StringVar()
        self.column_combo = ttk.Combobox(control_frame, textvariable=self.column_var, width=15)
        self.column_combo.pack(side=tk.LEFT, padx=5)
        
        # Populate columns
        self.update_column_choices()
        
        # Plot type selection
        ttk.Label(control_frame, text="Plot Type:").pack(side=tk.LEFT, padx=5)
        self.plot_type = tk.StringVar(value="line")
        plot_choices = ["line", "bar", "scatter"]
        plot_menu = ttk.Combobox(control_frame, textvariable=self.plot_type, values=plot_choices, width=8)
        plot_menu.pack(side=tk.LEFT, padx=5)
        
        # Update button
        ttk.Button(control_frame, text="Update Plot", command=self.create_plot).pack(side=tk.LEFT, padx=5)
        
        # Figure frame
        self.figure_frame = ttk.Frame(self.main_frame)
        self.figure_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for figure frame
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

    def update_column_choices(self):
        try:
            # Get numerical columns from the table
            columns = self.conn.execute(f"SELECT * FROM {self.selected_table} LIMIT 0").df().columns
            numerical_columns = [col for col in columns if 
                              col not in ['date', 'symbol', 'ticker', 'sector', 'industry'] and 
                              self.conn.execute(f"SELECT typeof({col}) FROM {self.selected_table} LIMIT 1").fetchone()[0]
                              in ['INTEGER', 'DOUBLE', 'FLOAT', 'DECIMAL']]
            
            self.column_combo['values'] = numerical_columns
            if numerical_columns:
                self.column_combo.set(numerical_columns[0])
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get columns: {e}")

    def get_time_delta(self):
        range_map = {
            "1M": timedelta(days=30),
            "3M": timedelta(days=90),
            "6M": timedelta(days=180),
            "1Y": timedelta(days=365),
            "3Y": timedelta(days=3*365),
            "5Y": timedelta(days=5*365),
            "ALL": None
        }
        return range_map[self.time_range.get()]

    def create_plot(self):
        try:
            # Clear previous plot
            for widget in self.figure_frame.winfo_children():
                widget.destroy()
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.set_tight_layout(True)
            
            # Get selected column and time range
            column = self.column_var.get()
            time_delta = self.get_time_delta()
            
            # Build query
            query = f"""
                SELECT date, symbol, {column}
                FROM {self.selected_table}
                WHERE symbol IN ({','.join(['?' for _ in self.selected_tickers])})
            """
            params = self.selected_tickers
            
            if time_delta:
                query += " AND date >= ?"
                params.append((datetime.now() - time_delta).strftime('%Y-%m-%d'))
            
            query += " ORDER BY date"
            
            # Execute query and plot data
            df = self.conn.execute(query, params).df()
            
            plot_type = self.plot_type.get()
            for ticker in self.selected_tickers:
                ticker_data = df[df['symbol'] == ticker]
                if plot_type == 'line':
                    ax.plot(ticker_data['date'], ticker_data[column], label=ticker)
                elif plot_type == 'bar':
                    ax.bar(ticker_data['date'], ticker_data[column], label=ticker, alpha=0.5)
                elif plot_type == 'scatter':
                    ax.scatter(ticker_data['date'], ticker_data[column], label=ticker)
            
            # Customize plot
            ax.set_xlabel('Date')
            ax.set_ylabel(column)
            ax.set_title(f'{column} for Selected Tickers')
            ax.legend()
            ax.grid(True)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Create canvas and add to frame
            canvas = FigureCanvasTkAgg(fig, master=self.figure_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create plot: {e}")

    def cleanup(self):
        """Cleanup resources"""
        try:
            self.conn.close()
            print("Database connection closed")
        except Exception as e:
            print(f"Error closing database connection: {e}") 