import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ticker_ai_agent import TickerAIAgent
import pandas as pd
from datetime import datetime, timedelta
from matplotlib.dates import DateFormatter
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import duckdb
import os

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
    def __init__(self, root, selected_tickers, selected_table, connection=None):
        """Initialize the plotter with optional existing connection"""
        self.root = root
        self.selected_tickers = selected_tickers
        self.selected_table = selected_table
        self.conn = connection
        self.ticker_column = None
        
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
    def __init__(self, root, selected_tickers, selected_table, selected_fields):
        # Initialize basic attributes first
        self.root = root
        self.selected_tickers = selected_tickers
        self.selected_table = selected_table
        self.selected_fields = selected_fields
        
        # Create database connection
        try:
            self.conn = duckdb.connect('historical_market_data.db', read_only=True)
            print("Successfully connected to database")
        except Exception as e:
            print(f"Database connection error: {e}")
            messagebox.showerror("Database Error", f"Failed to connect to database: {e}")
            raise
        
        try:
            # Create plot window
            self.plot_window = tk.Toplevel(root)
            self.plot_window.title("Predictions Plot")
            self.plot_window.geometry("1200x800")  # Increased window size
            
            # Create figure with subplots for each field
            self.create_plots()
            
            # Setup cleanup on window close
            self.plot_window.protocol("WM_DELETE_WINDOW", self.cleanup)
            
        except Exception as e:
            print(f"Error creating plots: {e}")
            if hasattr(self, 'conn'):
                self.conn.close()
            messagebox.showerror("Error", f"Failed to create plots: {str(e)}")
            raise

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
        """Create subplots for each field"""
        num_fields = len(self.selected_fields)
        self.fig = plt.figure(figsize=(12, 6*num_fields))
        gs = GridSpec(num_fields, 1, figure=self.fig)
        
        for i, field in enumerate(self.selected_fields):
            ax = self.fig.add_subplot(gs[i])
            
            for ticker in self.selected_tickers:
                try:
                    # Get historical data
                    df = self.get_historical_data(ticker, field)
                    if not df.empty:
                        # Plot historical data
                        ax.plot(df['date'], df[field], label=f'{ticker} Historical')
                        
                        # Create AI agent and get predictions
                        try:
                            ai_agent = TickerAIAgent(self.selected_table, connection=self.conn)
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
        
        plt.tight_layout()
        
        # Embed plot in tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_window)
        self.canvas.draw()
        
        # Add navigation toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_window)
        self.toolbar.update()
        
        # Pack canvas and toolbar
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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