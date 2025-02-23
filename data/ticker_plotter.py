import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import pandas as pd
from datetime import datetime, timedelta
from ticker_ai_agent import TickerAIAgent
import re

class TickerPlotter:
    def __init__(self, root, selected_tickers, selected_table, connection=None):
        """Initialize the plotter with optional existing connection"""
        self.root = root
        self.selected_tickers = selected_tickers
        self.selected_table = selected_table
        self.conn = connection
        
        if self.conn is None:
            raise ValueError("A valid database connection must be provided.")
        
        # Detect the correct ticker column
        self.ticker_column = self.detect_ticker_column()
        
        # Define fields based on table type
        self.table_fields = {
            'balance_sheets': ['total_assets', 'total_liabilities', 'total_equity', 'working_capital'],
            'financial_ratios': ['pe_ratio', 'price_to_book', 'debt_to_equity', 'current_ratio', 'quick_ratio', 'roe', 'roa', 'profit_margin'],
            'income_statements': ['total_revenue', 'gross_profit', 'operating_income', 'net_income', 'eps', 'ebitda'],
            'historical_prices': ['open', 'high', 'low', 'close', 'volume'],
            'market_sentiment': ['rsi', 'macd', 'bollinger_position', 'sentiment_score'],
            'stock_metrics': ['open_price', 'close_price', 'market_cap', 'pe_ratio', 'dividend_yield']
        }
        
        self.selected_fields = self.table_fields.get(selected_table, ['close'])
        
        # Create plot window
        self.plot_window = tk.Toplevel(self.root)
        self.plot_window.title(f"Market Analysis - {', '.join(selected_tickers)}")
        self.plot_window.geometry("1200x800")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.plot_window)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create metrics selection
        self.create_metrics_selector()
        
        # Initialize the figure and canvas
        self.fig = plt.figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.main_frame)
        self.toolbar.update()

    def detect_ticker_column(self):
        """Detect the ticker column in the selected table"""
        try:
            # Get column names
            columns = self.conn.execute(f"PRAGMA table_info({self.selected_table})").df()['name'].values
            print(f"Available columns: {columns}")
            
            # Look for common ticker column names
            if self.selected_table == 'historical_forex' and 'pair' in columns:
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

    def fetch_data(self):
        """Fetch data for selected tickers"""
        data = {}
        
        for ticker in self.selected_tickers:
            query = f"""
                SELECT date, {', '.join(self.selected_fields)}
                FROM {self.selected_table}
                WHERE {self.ticker_column} = ?
                ORDER BY date
            """
            df = self.conn.execute(query, [ticker]).df()
            df['date'] = pd.to_datetime(df['date'])
            data[ticker] = df
        return data

    def create_metrics_selector(self):
        """Create metrics selection frame"""
        metrics_frame = ttk.LabelFrame(self.main_frame, text="Metrics Selection", padding="5")
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create scrollable frame for metrics
        canvas = tk.Canvas(metrics_frame)
        scrollbar = ttk.Scrollbar(metrics_frame, orient="horizontal", command=canvas.xview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(xscrollcommand=scrollbar.set)
        
        self.metric_vars = {}
        for metric in self.selected_fields:
            var = tk.BooleanVar(value=True)
            self.metric_vars[metric] = var
            ttk.Checkbutton(scrollable_frame, text=metric.replace('_', ' ').title(), 
                           variable=var, command=self.update_plot).pack(side=tk.LEFT, padx=5)

        canvas.pack(side="top", fill="x", expand=True)
        scrollbar.pack(side="bottom", fill="x")

    def update_plot(self):
        """Update the plot based on selected metrics"""
        try:
            self.fig.clear()
            data = self.fetch_data()
            
            active_metrics = [m for m, v in self.metric_vars.items() if v.get()]
            if not active_metrics:
                return
                
            n_metrics = len(active_metrics)
            fig_rows = (n_metrics + 1) // 2  # Two metrics per row
            
            for i, metric in enumerate(active_metrics, 1):
                ax = self.fig.add_subplot(fig_rows, 2, i)
                
                for ticker in self.selected_tickers:
                    df = data[ticker]
                    ax.plot(df['date'], df[metric], label=ticker, marker='o')
                    
                ax.set_title(metric.replace('_', ' ').title())
                ax.set_xlabel('Date')
                ax.set_ylabel('Value')
                ax.grid(True)
                ax.legend()
                ax.tick_params(axis='x', rotation=45)
                
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"Error updating plot: {str(e)}")
            messagebox.showerror("Plot Error", f"Failed to update plot: {str(e)}")

    def create_plot(self):
        try:
            print("\nStarting plot creation...")
            # Clear previous plot
            for widget in self.figure_frame.winfo_children():
                widget.destroy()
            
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(10, 6))
            fig.set_tight_layout(True)
            
            # Get selected column and time range
            column = self.column_var.get()
            time_delta = self.get_time_delta()
            
            print(f"Selected column: {column}")
            print(f"Selected tickers: {self.selected_tickers}")
            print(f"Using ticker column: {self.ticker_column}")
            
            # For forex data, ensure we're using the correct format
            if self.selected_table == 'historical_forex':
                # Clean up forex pair format
                clean_tickers = [ticker.replace('=X', '').upper() for ticker in self.selected_tickers]
                
                # Build query for forex data
                query = f"""
                    SELECT date, pair as symbol, {column}
                    FROM {self.selected_table}
                    WHERE UPPER(REGEXP_REPLACE(pair, '[^A-Za-z]', '')) IN 
                        ({','.join(['?' for _ in clean_tickers])})
                """
                params = clean_tickers
            else:
                # Regular query for other tables
                query = f"""
                    SELECT date, {self.ticker_column} as symbol, {column}
                    FROM {self.selected_table}
                    WHERE {self.ticker_column} IN ({','.join(['?' for _ in self.selected_tickers])})
                """
                params = self.selected_tickers
            
            if time_delta:
                query += " AND date >= ?"
                params.append((datetime.now() - time_delta).strftime('%Y-%m-%d'))
            
            query += " ORDER BY date"
            
            print(f"Executing query: {query}")
            print(f"With parameters: {params}")
            
            # Execute query and plot data
            df = self.conn.execute(query, params).df()
            print(f"Retrieved {len(df)} rows of data")
            
            if df.empty:
                print("Warning: No data returned from query")
                messagebox.showwarning("No Data", "No data found for the selected parameters")
                return
            
            plot_type = self.plot_type.get()
            print(f"Creating {plot_type} plot")
            
            # For forex, we'll plot against the original ticker names
            ticker_map = {ticker.replace('=X', '').upper(): ticker for ticker in self.selected_tickers}
            
            for ticker in (self.selected_tickers if self.selected_table != 'historical_forex' else ticker_map.keys()):
                display_ticker = ticker if self.selected_table != 'historical_forex' else ticker_map[ticker]
                ticker_data = df[df['symbol'].str.replace('[^A-Za-z]', '', regex=True).str.upper() == ticker.replace('=X', '')]
                print(f"Plotting {len(ticker_data)} points for {display_ticker}")
                
                if len(ticker_data) == 0:
                    print(f"Warning: No data found for ticker {display_ticker}")
                    continue
                    
                if plot_type == 'line':
                    ax.plot(ticker_data['date'], ticker_data[column], label=display_ticker)
                elif plot_type == 'bar':
                    ax.bar(ticker_data['date'], ticker_data[column], label=display_ticker, alpha=0.5)
                elif plot_type == 'scatter':
                    ax.scatter(ticker_data['date'], ticker_data[column], label=display_ticker)
            
            # Customize plot
            ax.set_xlabel('Date')
            ax.set_ylabel(column)
            ax.set_title(f'{column} for Selected {self.ticker_column.title()}s')
            ax.legend()
            ax.grid(True)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            print("Creating canvas...")
            # Create canvas and add to frame
            canvas = FigureCanvasTkAgg(fig, master=self.figure_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            print("Plot creation completed successfully")
            
        except Exception as e:
            print(f"Error creating plot: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()
            messagebox.showerror("Error", f"Failed to create plot: {e}")

    def cleanup(self):
        """Cleanup resources and close window"""
        try:
            if hasattr(self, 'plot_window'):
                self.plot_window.destroy()
        except Exception as e:
            print(f"Error during cleanup: {e}")