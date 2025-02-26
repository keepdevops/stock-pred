import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ai_agent import TickerAIAgent
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import duckdb
import os
import tensorflow as tf
import numpy as np

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind('<Enter>', self.show_tooltip)
        self.widget.bind('<Leave>', self.hide_tooltip)

    def show_tooltip(self, event=None):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        # Create tooltip window
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = ttk.Label(self.tooltip, text=self.text, 
                         justify=tk.LEFT, background="#ffffe0", 
                         relief=tk.SOLID, borderwidth=1,
                         wraplength=300, padding=(5, 5))
        label.pack()

    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

class TickerPlotter:
    def __init__(self, root, selected_tickers, selected_table, connection=None, agent_class=None):
        """Initialize the plotter with optional existing connection"""
        self.root = root
        self.selected_tickers = selected_tickers
        self.selected_table = selected_table
        self.conn = connection
        self.ticker_column = None
        self.agent_class = agent_class
        
        # Create plot window
        self.plot_window = tk.Toplevel(self.root)
        self.plot_window.title(f"Predictions Plot - {', '.join(selected_tickers)}")
        self.plot_window.geometry("1200x800")
        
        # Initialize the figure and canvas
        self.fig = plt.figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_window)
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_window)
        self.toolbar.update()
        
        # Pack the canvas
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def detect_ticker_column(self):
        """Detect the ticker column in the selected table"""
        try:
            # Get column names
            cursor = self.conn.cursor()
            cursor.execute(f"PRAGMA table_info({self.selected_table})")
            columns = [col[1] for col in cursor.fetchall()]
            print(f"Available columns: {columns}")
            
            # Look for common ticker column names
            ticker_columns = ['symbol', 'ticker', 'pair']
            for col in ticker_columns:
                if col in columns:
                    self.ticker_column = col
                    print(f"Using '{col}' as ticker identifier")
                    return
                    
            print("No standard ticker column found")
            
        except Exception as e:
            print(f"Error detecting ticker column: {e}")
            raise

    def create_plot(self):
        """Create the plot with predictions"""
        try:
            print("\nStarting plot creation...")
            print(f"Current table: {self.selected_table}")
            
            # Detect ticker column if not already set
            if not self.ticker_column:
                self.detect_ticker_column()
            
            print(f"Ticker column: {self.ticker_column}")
            
            # Create the AI agent with the existing connection
            self.ai_agent = TickerAIAgent(
                tickers=self.selected_tickers,
                fields=['sector', 'fiscal_year', 'total_assets', 'total_liabilities'],
                model_type='lstm',
                parameters={
                    'units': 50,
                    'dropout': 0.2,
                    'learning_rate': 0.001,
                    'epochs': 100
                },
                connection=self.conn
            )
            
            print("\nStarting prediction process:")
            print(f"Selected tickers: {self.selected_tickers}")
            print(f"Selected fields: {self.ai_agent.fields}")
            print(f"Model type: {self.ai_agent.model_type}")
            print(f"Parameters: Units={self.ai_agent.parameters['units']}, "
                  f"Dropout={self.ai_agent.parameters['dropout']}, "
                  f"Learning Rate={self.ai_agent.parameters['learning_rate']}, "
                  f"Epochs={self.ai_agent.parameters['epochs']}")
            
            # Train models and make predictions
            self.ai_agent.train_and_predict()
            
        except Exception as e:
            print(f"Error creating plot: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            messagebox.showerror("Plot Error", f"Failed to create plot: {str(e)}")

    def cleanup(self):
        """Cleanup resources and close window"""
        try:
            if hasattr(self, 'plot_window'):
                self.plot_window.destroy()
        except Exception as e:
            print(f"Error during cleanup: {e}")

class PredictionsPlotter:
    def __init__(self, model, data, prediction_days=30):
        self.model = model
        self.data = data
        self.prediction_days = prediction_days

    def make_predictions(self):
        predictions = self.model.predict(self.data)
        self.plot_predictions(predictions)

    def plot_predictions(self, predictions):
        plt.figure(figsize=(10, 6))
        plt.plot(predictions, label='Predictions', linestyle='--', color='r')
        plt.title('Model Predictions')
        plt.xlabel('Time')
        plt.ylabel('Predicted Values')
        plt.legend()
        plt.show()

    def extend_predictions(self):
        # Extend predictions for future days
        future_data = np.random.rand(self.prediction_days, self.data.shape[1])  # Example future data
        future_predictions = self.model.predict(future_data)
        return future_predictions

    def get_historical_data(self, ticker, field):
        """Get historical data for a specific ticker and field"""
        try:
            # Get the correct ticker column name
            columns = self.conn.execute(f"SELECT * FROM {self.selected_table} LIMIT 0").df().columns
            ticker_column = None

            # Force 'pair' column for historical_forex tables
            if self.selected_table == 'historical_forex' and 'pair' in columns:
                ticker_column = 'pair'
                print("Using 'pair' column for historical_forex table.")
            else:
                # Check for possible ticker columns in order of preference
                for col in ['symbol', 'ticker', 'pair']:
                    if col in columns:
                        ticker_column = col
                        break

            if not ticker_column:
                raise ValueError(f"No ticker column found in table {self.selected_table}")

            query = f"""
                SELECT date, {field}
                FROM {self.selected_table}
                WHERE {ticker_column} = ?
                    AND {field} IS NOT NULL
                    AND CAST({field} AS VARCHAR) != ''
                ORDER BY date ASC
            """
            return self.conn.execute(query, [ticker]).df()
        except Exception as e:
            print(f"Error getting historical data: {e}")
            return pd.DataFrame()

    def create_plots(self):
        """Create individual plots for each field in separate frames"""
        for field in self.selected_fields:
            # Create a new frame for each field
            field_frame = ttk.LabelFrame(self.plot_window, text=f"{field.replace('_', ' ').title()} Plot", padding="5")
            field_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create a figure for the current field
            fig = plt.Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            
            for ticker in self.selected_tickers:
                try:
                    # Get historical data
                    df = self.get_historical_data(ticker, field)
                    if not df.empty:
                        # Plot historical data
                        ax.plot(df['date'], df[field], label=f'{ticker} Historical')
                        
                        # Create AI agent and get predictions
                        try:
                            ai_agent = self.agent_class(self.selected_table, connection=self.conn, model_type=self.ai_model_type)
                            # Train the model if it doesn't exist
                            model_path = f'models/{ticker}_{field}_lstm_model.keras'
                            if not os.path.exists(model_path):
                                ai_agent.train(ticker, field)
                            
                            # Get predictions
                            future_dates, predictions = ai_agent.predict(ticker, field)
                            ax.plot(future_dates, predictions, '--', label=f'{ticker} Predicted')
                        except Exception as e:
                            print(f"Could not plot predictions for {ticker}-{field}: {e}")
                
                except Exception as e:
                    print(f"Error plotting {ticker}-{field}: {e}")
                    continue
            
            ax.set_title(f'{field.replace("_", " ").title()} Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel(field.replace("_", " ").title())
            ax.legend()
            ax.grid(True)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45)
            
            # Embed plot in tkinter window
            canvas = FigureCanvasTkAgg(fig, master=field_frame)
            canvas.draw()
            
            # Add navigation toolbar
            toolbar = NavigationToolbar2Tk(canvas, field_frame)
            toolbar.update()
            
            # Pack canvas and toolbar
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def show_model_help(self):
        """Show help dialog with model descriptions"""
        help_text = """
Model Types and Their Characteristics:

LSTM (Long Short-Term Memory)
- Best for: General time series prediction
- Strengths: Handles long-term dependencies well
- Use when: You have a standard time series prediction task

GRU (Gated Recurrent Unit)
- Best for: Similar to LSTM but faster training
- Strengths: More efficient, fewer parameters
- Use when: You want faster training with similar performance

Simple
- Best for: Quick experiments and simpler patterns
- Strengths: Fast training, less prone to overfitting
- Use when: You have limited data or need quick results

Deep
- Best for: Complex patterns and relationships
- Strengths: Can capture intricate patterns
- Use when: You have lots of data and complex patterns

Bidirectional
- Best for: Patterns that depend on future context
- Strengths: Can use both past and future information
- Use when: The full sequence context is important

Transformer
- Best for: Capturing global dependencies
- Strengths: Excellent at handling long-range patterns
- Use when: You have long sequences with important relationships

CNN-LSTM
- Best for: Local pattern recognition with temporal features
- Strengths: Good at detecting local patterns and trends
- Use when: Your data has both local and temporal patterns

Dual LSTM
- Best for: Multiple time scales
- Strengths: Can capture both short and long-term patterns
- Use when: Your data has multiple temporal scales

Attention LSTM
- Best for: Focusing on important time steps
- Strengths: Can learn which parts of sequence are important
- Use when: Not all time steps are equally important

Hybrid
- Best for: Complex data with multiple pattern types
- Strengths: Combines benefits of multiple approaches
- Use when: Your data has various types of patterns

Parameters:
- Units: Number of neurons in each layer
- Dropout: Regularization rate (0-1)
- Learning Rate: Step size for optimization
- Batch Size: Number of samples per training batch
- Epochs: Number of training iterations
- Optimizer: Training optimization algorithm
"""
        
        help_window = tk.Toplevel(self.plot_window)
        help_window.title("Model Help")
        help_window.geometry("600x800")
        
        # Create text widget
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(help_window, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.configure(yscrollcommand=scrollbar.set)
        
        # Insert help text
        text_widget.insert(tk.END, help_text)
        text_widget.configure(state='disabled')  # Make read-only

    def cleanup(self):
        """Cleanup resources and close window"""
        try:
            if hasattr(self, 'conn'):
                self.conn.close()
                print("Database connection closed")
            if hasattr(self, 'plot_window'):
                self.plot_window.destroy()
        except Exception as e:
            print(f"Error during cleanup: {e}")

    def update_plot(self):
        """Update the plot based on selected metrics"""
        try:
            self.fig.clear()
            data = self.fetch_data()
            
            active_metrics = [m for m, v in self.metric_vars.items() if v.get()]
            if not active_metrics:
                print("No active metrics selected for plotting.")
                return
                
            n_metrics = len(active_metrics)
            fig_rows = (n_metrics + 1) // 2  # Two metrics per row
            
            for i, metric in enumerate(active_metrics, 1):
                ax = self.fig.add_subplot(fig_rows, 2, i)
                
                for ticker in self.selected_tickers:
                    df = data[ticker]
                    if df.empty:
                        print(f"No data to plot for {ticker} and metric {metric}")
                        continue
                    ax.plot(df['date'], df[metric], label=ticker, marker='o')
                    print(f"Plotting {len(df)} points for {ticker} - {metric}")
                    
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
            print(f"Fetched {len(df)} rows for {ticker}")
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
            var = tk.BooleanVar(value=True)  # Default to True to ensure at least one metric is selected
            self.metric_vars[metric] = var
            ttk.Checkbutton(scrollable_frame, text=metric.replace('_', ' ').title(), 
                           variable=var, command=self.update_plot).pack(side=tk.LEFT, padx=5)

        canvas.pack(side="top", fill="x", expand=True)
        scrollbar.pack(side="bottom", fill="x")

    def create_plots(self):
        """Create individual plots for each field in separate frames"""
        for field in self.selected_fields:
            # Create a new frame for each field
            field_frame = ttk.LabelFrame(self.plot_window, text=f"{field.replace('_', ' ').title()} Plot", padding="5")
            field_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create a figure for the current field
            fig = plt.Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            
            for ticker in self.selected_tickers:
                try:
                    # Get historical data
                    df = self.get_historical_data(ticker, field)
                    if not df.empty:
                        # Plot historical data
                        ax.plot(df['date'], df[field], label=f'{ticker} Historical')
                        
                        # Create AI agent and get predictions
                        try:
                            ai_agent = self.agent_class(self.selected_table, connection=self.conn, model_type=self.ai_model_type)
                            # Train the model if it doesn't exist
                            model_path = f'models/{ticker}_{field}_lstm_model.keras'
                            if not os.path.exists(model_path):
                                ai_agent.train(ticker, field)
                            
                            # Get predictions
                            future_dates, predictions = ai_agent.predict(ticker, field)
                            ax.plot(future_dates, predictions, '--', label=f'{ticker} Predicted')
                        except Exception as e:
                            print(f"Could not plot predictions for {ticker}-{field}: {e}")
                
                except Exception as e:
                    print(f"Error plotting {ticker}-{field}: {e}")
                    continue
            
            ax.set_title(f'{field.replace("_", " ").title()} Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel(field.replace("_", " ").title())
            ax.legend()
            ax.grid(True)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45)
            
            # Embed plot in tkinter window
            canvas = FigureCanvasTkAgg(fig, master=field_frame)
            canvas.draw()
            
            # Add navigation toolbar
            toolbar = NavigationToolbar2Tk(canvas, field_frame)
            toolbar.update()
            
            # Pack canvas and toolbar
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def create_plots(self):
        """Create individual plots for each field in separate frames"""
        for field in self.selected_fields:
            # Create a new frame for each field
            field_frame = ttk.LabelFrame(self.plot_window, text=f"{field.replace('_', ' ').title()} Plot", padding="5")
            field_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Create a figure for the current field
            fig = plt.Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            
            for ticker in self.selected_tickers:
                try:
                    # Get historical data
                    df = self.get_historical_data(ticker, field)
                    if not df.empty:
                        # Plot historical data
                        ax.plot(df['date'], df[field], label=f'{ticker} Historical')
                        
                        # Create AI agent and get predictions
                        try:
                            ai_agent = self.agent_class(self.selected_table, connection=self.conn, model_type=self.ai_model_type)
                            # Train the model if it doesn't exist
                            model_path = f'models/{ticker}_{field}_lstm_model.keras'
                            if not os.path.exists(model_path):
                                ai_agent.train(ticker, field)
                            
                            # Get predictions
                            future_dates, predictions = ai_agent.predict(ticker, field)
                            ax.plot(future_dates, predictions, '--', label=f'{ticker} Predicted')
                        except Exception as e:
                            print(f"Could not plot predictions for {ticker}-{field}: {e}")
                
                except Exception as e:
                    print(f"Error plotting {ticker}-{field}: {e}")
                    continue
            
            ax.set_title(f'{field.replace("_", " ").title()} Over Time')
            ax.set_xlabel('Date')
            ax.set_ylabel(field.replace("_", " ").title())
            ax.legend()
            ax.grid(True)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45)
            
            # Embed plot in tkinter window
            canvas = FigureCanvasTkAgg(fig, master=field_frame)
            canvas.draw()
            
            # Add navigation toolbar
            toolbar = NavigationToolbar2Tk(canvas, field_frame)
            toolbar.update()
            
            # Pack canvas and toolbar
            canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True) 