"""
Visualization panel for the application
"""
import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import tkinter.messagebox as messagebox

class VisualizationPanel(ttk.Frame):
    def __init__(self, parent):
        """Initialize the visualization panel"""
        super().__init__(parent)
        self.parent = parent
        
        # Create notebook for multiple visualizations
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        
        # Create tabs
        self.historical_tab = ttk.Frame(self.notebook)
        self.training_tab = ttk.Frame(self.notebook)
        self.prediction_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.historical_tab, text="Historical Data")
        self.notebook.add(self.training_tab, text="Training Results")
        self.notebook.add(self.prediction_tab, text="Predictions")
        
        # Initialize the historical tab
        self._init_historical_tab()
        
        # Initialize the training tab
        self._init_training_tab()
        
        # Initialize the prediction tab
        self.initialize_prediction_tab()
        
        print("Visualization panel initialized")
        
    def _init_historical_tab(self):
        """Initialize the historical data tab"""
        # Create a frame for controls
        self.historical_control_frame = ttk.Frame(self.historical_tab)
        self.historical_control_frame.pack(fill="x", padx=5, pady=5)
        
        # Ticker selection for historical data
        ttk.Label(self.historical_control_frame, text="Ticker:").pack(side="left", padx=5)
        self.historical_ticker_var = tk.StringVar()
        self.historical_ticker_combo = ttk.Combobox(self.historical_control_frame, 
                                                  textvariable=self.historical_ticker_var,
                                                  state="readonly", width=10)
        self.historical_ticker_combo.pack(side="left", padx=5)
        self.historical_ticker_combo.bind("<<ComboboxSelected>>", self._on_historical_ticker_selected)
        
        # View type selection
        ttk.Label(self.historical_control_frame, text="View:").pack(side="left", padx=5)
        self.historical_view_var = tk.StringVar(value="Price")
        view_combo = ttk.Combobox(self.historical_control_frame, 
                                textvariable=self.historical_view_var,
                                state="readonly", width=15,
                                values=["Price", "Volume", "Price & Volume", "Statistics"])
        view_combo.pack(side="left", padx=5)
        view_combo.bind("<<ComboboxSelected>>", self._on_historical_view_selected)
        
        # Create a frame for the matplotlib figure
        self.historical_frame = ttk.Frame(self.historical_tab)
        self.historical_frame.pack(fill="both", expand=True)
        
        # Create figure and canvas
        self.historical_fig = Figure(figsize=(10, 6), dpi=100)
        self.historical_canvas = FigureCanvasTkAgg(self.historical_fig, self.historical_frame)
        self.historical_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add toolbar
        self.historical_toolbar = NavigationToolbar2Tk(self.historical_canvas, self.historical_frame)
        self.historical_toolbar.update()
        
        # Create a frame for the data table (initially hidden)
        self.table_frame = ttk.Frame(self.historical_tab)
        
        # Create treeview for data table
        self.data_table = ttk.Treeview(self.table_frame)
        
        # Add vertical scrollbar
        vsb = ttk.Scrollbar(self.table_frame, orient="vertical", command=self.data_table.yview)
        self.data_table.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        
        # Add horizontal scrollbar
        hsb = ttk.Scrollbar(self.table_frame, orient="horizontal", command=self.data_table.xview)
        self.data_table.configure(xscrollcommand=hsb.set)
        hsb.pack(side="bottom", fill="x")
        
        # Pack the treeview
        self.data_table.pack(fill="both", expand=True)
        
        # Initial message
        ax = self.historical_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Select a ticker to view historical data", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        self.historical_canvas.draw()
        
        # Store historical data
        self.historical_data = {}
        
    def _toggle_view_mode(self):
        """Toggle between chart and table views"""
        current_ticker = self.historical_ticker_var.get()
        if not current_ticker or current_ticker not in self.historical_data:
            messagebox.showinfo("No Data", "Please select a ticker with data first.")
            return
            
        if self.view_mode_var.get() == "Chart":
            # Switch to table view
            self.view_mode_var.set("Table")
            self.toggle_view_btn.config(text="Show Chart")
            
            # Hide chart frame and show table frame
            self.historical_frame.pack_forget()
            self.table_frame.pack(fill="both", expand=True)
            
            # Display data in table
            self._display_data_in_table(current_ticker)
        else:
            # Switch to chart view
            self.view_mode_var.set("Chart")
            self.toggle_view_btn.config(text="Show Table")
            
            # Hide table frame and show chart frame
            self.table_frame.pack_forget()
            self.historical_frame.pack(fill="both", expand=True)
            
    def _display_data_in_table(self, ticker):
        """Display data for the selected ticker in the table view"""
        if ticker not in self.historical_data:
            return
            
        df = self.historical_data[ticker]
        column_mapping = self._get_column_mapping(df)
        
        # Clear existing data
        self.data_table.delete(*self.data_table.get_children())
        
        # Clear existing columns
        for col in self.data_table['columns']:
            self.data_table.heading(col, text="")
            self.data_table.column(col, width=0)
        
        # Determine which data to display based on format
        if column_mapping.get('format') == 'financial_statement':
            self._display_financial_data_in_table(ticker, df)
        elif column_mapping.get('format') == 'sentiment':
            self._display_sentiment_data_in_table(ticker, df, column_mapping)
        else:
            # Time series data (price, volume, etc.)
            self._display_timeseries_data_in_table(ticker, df, column_mapping)
    
    def _display_timeseries_data_in_table(self, ticker, df, column_mapping):
        """Display time series data in table view"""
        # Sort by date
        date_col = column_mapping['date']
        df = df.sort_values(by=date_col, ascending=False).copy()  # Show most recent first
        
        # Create columns
        columns = [col for col in df.columns if col != '#0']  # Skip the first column (Treeview uses #0)
        self.data_table['columns'] = columns
        
        # Configure columns
        self.data_table.heading('#0', text="Index")
        self.data_table.column('#0', width=60)
        
        for col in columns:
            self.data_table.heading(col, text=col.capitalize())
            if col == date_col:
                self.data_table.column(col, width=100)
            else:
                self.data_table.column(col, width=80)
        
        # Add data rows (limit to most recent 100 to avoid performance issues)
        for i, (_, row) in enumerate(df.head(100).iterrows()):
            values = [row[col] for col in columns]
            self.data_table.insert('', 'end', text=str(i+1), values=values)
    
    def _display_financial_data_in_table(self, ticker, df):
        """Display financial statement data in table view"""
        # Filter for this ticker
        ticker_data = df[df['ticker'] == ticker].copy()
        
        # Create pivot table
        try:
            # Get unique dates and sort them
            dates = sorted(ticker_data['date'].unique())
            
            # If there are too many dates, use the most recent ones
            if len(dates) > 10:
                dates = dates[-10:]  # Last 10 dates
                ticker_data = ticker_data[ticker_data['date'].isin(dates)]
            
            # Create pivot table
            pivot_data = ticker_data.pivot_table(
                index='metric', 
                columns='date', 
                values='value',
                aggfunc='first'  # In case of duplicates
            )
            
            # Reset index to make metric a column
            pivot_data = pivot_data.reset_index()
            
            # Create columns for the table
            columns = ['metric'] + list(map(str, dates))
            self.data_table['columns'] = columns
            
            # Configure columns
            self.data_table.heading('#0', text="")
            self.data_table.column('#0', width=0, stretch=False)
            
            self.data_table.heading('metric', text="Metric")
            self.data_table.column('metric', width=150)
            
            for date in dates:
                self.data_table.heading(str(date), text=str(date))
                self.data_table.column(str(date), width=100)
            
            # Add data rows
            for _, row in pivot_data.iterrows():
                metric = row['metric']
                formatted_metric = metric.replace('_', ' ').title()
                values = [formatted_metric] + [self._format_value(row[str(date)]) for date in dates]
                self.data_table.insert('', 'end', text="", values=values)
                
        except Exception as e:
            print(f"Error creating financial data table: {e}")
            # Create a simple view of the raw data instead
            self._display_raw_data_in_table(ticker_data)
    
    def _display_sentiment_data_in_table(self, ticker, df, column_mapping):
        """Display sentiment data in table view"""
        # Filter for this ticker and sort by date
        ticker_data = df[df['ticker'] == ticker].sort_values(by=column_mapping['date'], ascending=False).copy()
        
        # Create columns
        columns = [col for col in ticker_data.columns if col != '#0']
        self.data_table['columns'] = columns
        
        # Configure columns
        self.data_table.heading('#0', text="")
        self.data_table.column('#0', width=0, stretch=False)
        
        for col in columns:
            # Format column headers to be more readable
            header = col.replace('_', ' ').title()
            self.data_table.heading(col, text=header)
            
            if col == column_mapping['date']:
                self.data_table.column(col, width=100)
            else:
                self.data_table.column(col, width=80)
        
        # Add data rows
        for _, row in ticker_data.iterrows():
            values = [row[col] for col in columns]
            self.data_table.insert('', 'end', text="", values=values)
    
    def _display_raw_data_in_table(self, df):
        """Display raw dataframe in table view as a fallback"""
        # Create columns
        columns = list(df.columns)
        self.data_table['columns'] = columns
        
        # Configure columns
        self.data_table.heading('#0', text="Index")
        self.data_table.column('#0', width=60)
        
        for col in columns:
            self.data_table.heading(col, text=col.capitalize())
            self.data_table.column(col, width=100)
        
        # Add data rows
        for i, (_, row) in enumerate(df.head(100).iterrows()):
            values = [row[col] for col in columns]
            self.data_table.insert('', 'end', text=str(i+1), values=values)
    
    def _format_value(self, value):
        """Format numeric values for display in tables"""
        if pd.isna(value):
            return "N/A"
        elif isinstance(value, (int, float)):
            if abs(value) >= 1e9:
                return f"${value/1e9:.2f}B"
            elif abs(value) >= 1e6:
                return f"${value/1e6:.2f}M"
            elif abs(value) >= 1e3:
                return f"${value/1e3:.2f}K"
            else:
                return f"${value:.2f}"
        else:
            return str(value)
        
    def _init_training_tab(self):
        """Initialize the training results tab"""
        # Create a frame for controls
        self.training_control_frame = ttk.Frame(self.training_tab)
        self.training_control_frame.pack(fill="x", padx=5, pady=5)
        
        # Ticker selection for training results
        ttk.Label(self.training_control_frame, text="Ticker:").pack(side="left", padx=5)
        self.training_ticker_var = tk.StringVar()
        self.training_ticker_combo = ttk.Combobox(self.training_control_frame, 
                                                 textvariable=self.training_ticker_var,
                                                 state="readonly", width=10)
        self.training_ticker_combo.pack(side="left", padx=5)
        self.training_ticker_combo.bind("<<ComboboxSelected>>", self._on_training_ticker_selected)
        
        # Create a frame for the matplotlib figure
        self.training_figure_frame = ttk.Frame(self.training_tab)
        self.training_figure_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create figure and canvas
        self.training_fig = Figure(figsize=(10, 6), dpi=100)
        self.training_canvas = FigureCanvasTkAgg(self.training_fig, self.training_figure_frame)
        self.training_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add toolbar
        self.training_toolbar = NavigationToolbar2Tk(self.training_canvas, self.training_figure_frame)
        self.training_toolbar.update()
        
        # Initial message
        ax = self.training_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Train a model to view results", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        self.training_canvas.draw()
        
        # Store training results
        self.training_results = {}
        
    def _on_historical_ticker_selected(self, event):
        """Handle historical ticker selection"""
        # Debugging output
        ticker = self.historical_ticker_var.get()
        print(f"Ticker selected: {ticker}")
        print(f"Available tickers in historical_data: {list(self.historical_data.keys())}")
        
        if ticker in self.historical_data:
            print(f"Plotting data for {ticker}")
            self._plot_historical_data(ticker)
        else:
            print(f"No data available for {ticker}, attempting to load it")
            # Show loading message
            self.historical_fig.clear()
            ax = self.historical_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Loading data for {ticker}...", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_axis_off()
            self.historical_canvas.draw()
            
            # Try to load data on-demand by notifying parent
            try:
                # Find main window and request data for this ticker
                parent = self.parent
                while parent:
                    if hasattr(parent, 'load_ticker_data'):
                        # Request data and get result
                        success = parent.load_ticker_data(ticker)
                        if success:
                            # Data should now be in historical_data, so plot it
                            if ticker in self.historical_data:
                                self._plot_historical_data(ticker)
                            return
                        break
                    try:
                        parent = parent.master
                    except:
                        break
                
                # If we get here, we couldn't load the data
                self.historical_fig.clear()
                ax = self.historical_fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Could not load data for {ticker}.\nPlease make sure the data exists in the selected database.", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_axis_off()
                self.historical_canvas.draw()
            except Exception as e:
                print(f"Error loading data for {ticker}: {e}")
                # Show error message in the plot area
                self.historical_fig.clear()
                ax = self.historical_fig.add_subplot(111)
                ax.text(0.5, 0.5, f"Error loading data for {ticker}.\n{str(e)}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, fontsize=12)
                ax.set_axis_off()
                self.historical_canvas.draw()
            
    def _on_historical_view_selected(self, event):
        """Handle historical view selection"""
        # Debugging output
        view_type = self.historical_view_var.get()
        ticker = self.historical_ticker_var.get()
        print(f"View selected: {view_type} for ticker {ticker}")
        
        if ticker in self.historical_data:
            print(f"Plotting {view_type} view for {ticker}")
            self._plot_historical_data(ticker)
        else:
            print(f"No data available for {ticker}")
            
    def _on_plot_data_clicked(self):
        """Handle Plot Data button click"""
        ticker = self.historical_ticker_var.get()
        if not ticker:
            messagebox.showinfo("No Ticker Selected", "Please select a ticker to plot data.")
            return
            
        if ticker in self.historical_data:
            self._plot_historical_data(ticker)
        else:
            messagebox.showerror("Data Not Available", f"No data available for {ticker}.")
            
    def _plot_historical_data(self, ticker):
        """Plot historical data based on selected view"""
        if ticker not in self.historical_data:
            return
            
        # Get the data
        df = self.historical_data[ticker]
        
        # Map standard column names to actual columns in the dataframe
        column_mapping = self._get_column_mapping(df)
        
        # Clear previous plot
        self.historical_fig.clear()
        
        # Make sure historical frame is visible (in case it was hidden)
        self.historical_frame.pack(fill="both", expand=True)
        if self.table_frame.winfo_ismapped():
            self.table_frame.pack_forget()
        
        # Check data format and route to appropriate plotting function
        if column_mapping.get('format') == 'financial_statement':
            self._plot_financial_statement_data(ticker, df, column_mapping)
            self.historical_canvas.draw()
            return
            
        if column_mapping.get('format') == 'sentiment':
            self._plot_sentiment_data(ticker, df, column_mapping)
            self.historical_canvas.draw()
            return
            
        if not column_mapping.get('date'):
            messagebox.showerror("Data Format Error", 
                                 f"Cannot find date column in data for {ticker}.")
            return
            
        # Get the selected view
        view_type = self.historical_view_var.get()
        
        if view_type == "Price" and column_mapping.get('price'):
            self._plot_price_data(ticker, df, column_mapping)
        elif view_type == "Volume" and column_mapping.get('volume'):
            self._plot_volume_data(ticker, df, column_mapping)
        elif view_type == "Price & Volume" and column_mapping.get('price') and column_mapping.get('volume'):
            self._plot_price_and_volume_data(ticker, df, column_mapping)
        elif view_type == "Statistics" and column_mapping.get('price'):
            self._plot_statistics_data(ticker, df, column_mapping)
        else:
            # Display message if the selected view is not available
            ax = self.historical_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Cannot display {view_type} view - required data columns missing.", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_axis_off()
            
        # Draw canvas
        self.historical_canvas.draw()
    
    def _get_column_mapping(self, df):
        """Map standard column names to actual columns in the dataframe"""
        columns = df.columns
        mapping = {}
        
        # Print available columns for debugging
        print(f"Available columns in dataframe: {columns}")
        
        # Check if this is financial statement data in long format
        if set(['ticker', 'date', 'metric', 'value']).issubset(set(columns)):
            mapping['format'] = 'financial_statement'
            mapping['date'] = 'date'
            mapping['metric'] = 'metric'
            mapping['value'] = 'value'
            return mapping
            
        # Check if this is sentiment data
        if set(['ticker', 'date']).issubset(set(columns)) and any(col in columns for col in ['sentiment', 'sentiment_score', 'score', 'positive', 'negative', 'neutral']):
            mapping['format'] = 'sentiment'
            mapping['date'] = 'date'
            
            # Map sentiment column
            if 'sentiment' in columns:
                mapping['sentiment'] = 'sentiment'
            elif 'sentiment_score' in columns:
                mapping['sentiment'] = 'sentiment_score'
            elif 'score' in columns:
                mapping['sentiment'] = 'score'
                
            # Map additional sentiment metrics if available
            if 'positive' in columns:
                mapping['positive'] = 'positive'
            if 'negative' in columns:
                mapping['negative'] = 'negative'
            if 'neutral' in columns:
                mapping['neutral'] = 'neutral'
            if 'volume' in columns:
                mapping['volume'] = 'volume'
                
            return mapping
        
        # Map date column
        if 'date' in columns:
            mapping['date'] = 'date'
        elif 'Date' in columns:
            mapping['date'] = 'Date'
        elif 'timestamp' in columns:
            mapping['date'] = 'timestamp'
        elif 'Timestamp' in columns:
            mapping['date'] = 'Timestamp'
        
        # Map price columns
        if 'close' in columns:
            mapping['price'] = 'close'
            mapping['high'] = 'high' if 'high' in columns else None
            mapping['low'] = 'low' if 'low' in columns else None
            mapping['open'] = 'open' if 'open' in columns else None
        elif 'Close' in columns:
            mapping['price'] = 'Close'
            mapping['high'] = 'High' if 'High' in columns else None
            mapping['low'] = 'Low' if 'Low' in columns else None
            mapping['open'] = 'Open' if 'Open' in columns else None
        elif 'price' in columns:
            mapping['price'] = 'price'
        elif 'Price' in columns:
            mapping['price'] = 'Price'
        elif 'settlement' in columns:  # For futures data
            mapping['price'] = 'settlement'
        elif 'Settlement' in columns:
            mapping['price'] = 'Settlement'
        elif 'Last' in columns:
            mapping['price'] = 'Last'
        elif 'last' in columns:
            mapping['price'] = 'last'
        
        # Map volume column
        if 'volume' in columns:
            mapping['volume'] = 'volume'
        elif 'Volume' in columns:
            mapping['volume'] = 'Volume'
        
        # Default format is time series
        mapping['format'] = 'time_series'
        return mapping
        
    def _plot_price_data(self, ticker, df, column_mapping):
        """Plot price data"""
        # Create subplot
        ax = self.historical_fig.add_subplot(111)
        
        date_col = column_mapping['date']
        price_col = column_mapping['price']
        
        # Plot closing/settlement prices
        ax.plot(df[date_col], df[price_col], label='Price', color='blue', linewidth=2)
        
        # Plot high and low as a light fill if available
        if column_mapping.get('high') and column_mapping.get('low'):
            ax.fill_between(df[date_col], df[column_mapping['low']], df[column_mapping['high']], 
                           alpha=0.2, color='blue')
        
        # Add title and labels
        data_type = "Price"
        if price_col in ['settlement', 'Settlement']:
            data_type = "Settlement Price"
        
        ax.set_title(f'{ticker} - Historical {data_type} Data')
        ax.set_xlabel('Date')
        ax.set_ylabel(data_type)
        ax.grid(True)
        ax.legend()
        
        # Rotate date labels
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Tight layout
        self.historical_fig.tight_layout()
        
    def _plot_volume_data(self, ticker, df, column_mapping):
        """Plot volume data"""
        # Create subplot
        ax = self.historical_fig.add_subplot(111)
        
        date_col = column_mapping['date']
        volume_col = column_mapping['volume']
        
        # Plot volume
        ax.bar(df[date_col], df[volume_col], alpha=0.6, color='purple', label='Volume')
        
        # Add title and labels
        ax.set_title(f'{ticker} - Historical Volume Data')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volume')
        ax.grid(True)
        ax.legend()
        
        # Rotate date labels
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Tight layout
        self.historical_fig.tight_layout()
        
    def _plot_price_and_volume_data(self, ticker, df, column_mapping):
        """Plot price and volume data"""
        # Create subplots
        ax1 = self.historical_fig.add_subplot(211)
        ax2 = self.historical_fig.add_subplot(212, sharex=ax1)
        
        date_col = column_mapping['date']
        price_col = column_mapping['price']
        volume_col = column_mapping['volume']
        
        # Plot closing prices on the first subplot
        ax1.plot(df[date_col], df[price_col], label='Price', color='blue', linewidth=2)
        
        # Plot high and low as a light fill if available
        if column_mapping.get('high') and column_mapping.get('low'):
            ax1.fill_between(df[date_col], df[column_mapping['low']], df[column_mapping['high']], 
                            alpha=0.2, color='blue')
        
        # Add title and labels
        data_type = "Price"
        if price_col in ['settlement', 'Settlement']:
            data_type = "Settlement Price"
            
        ax1.set_title(f'{ticker} - Historical {data_type} and Volume Data')
        ax1.set_ylabel(data_type)
        ax1.grid(True)
        ax1.legend()
        
        # Plot volume on the second subplot
        ax2.bar(df[date_col], df[volume_col], alpha=0.6, color='purple', label='Volume')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        ax2.legend()
        
        # Rotate date labels
        plt.setp(ax2.get_xticklabels(), rotation=45)
        
        # Tight layout
        self.historical_fig.tight_layout()
        
    def _plot_statistics_data(self, ticker, df, column_mapping):
        """Plot statistics about the data"""
        # Create a figure with two subplots
        ax1 = self.historical_fig.add_subplot(211)
        ax2 = self.historical_fig.add_subplot(212)
        
        date_col = column_mapping['date']
        price_col = column_mapping['price']
        
        # Calculate statistics
        price_mean = df[price_col].mean()
        price_std = df[price_col].std()
        price_min = df[price_col].min()
        price_max = df[price_col].max()
        price_median = df[price_col].median()
        
        # Calculate daily returns
        df['return'] = df[price_col].pct_change() * 100
        
        # First subplot: Histogram of prices
        ax1.hist(df[price_col], bins=30, alpha=0.7, color='blue')
        ax1.axvline(price_mean, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {price_mean:.2f}')
        ax1.axvline(price_median, color='green', linestyle='dashed', linewidth=2, label=f'Median: {price_median:.2f}')
        
        data_type = "Price"
        if price_col in ['settlement', 'Settlement']:
            data_type = "Settlement Price"
            
        ax1.set_title(f'{ticker} - {data_type} Distribution')
        ax1.set_xlabel(data_type)
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True)
        
        # Second subplot: Daily returns
        ax2.hist(df['return'].dropna(), bins=30, alpha=0.7, color='green')
        ax2.axvline(0, color='red', linestyle='dashed', linewidth=2)
        ax2.set_title(f'{ticker} - Daily Returns Distribution')
        ax2.set_xlabel('Daily Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.grid(True)
        
        # Create text with statistics
        stats_text = f"Statistics for {ticker}:\n"
        stats_text += f"Number of samples: {len(df)}\n"
        stats_text += f"Date range: {df[date_col].min()} to {df[date_col].max()}\n"
        stats_text += f"Mean {data_type.lower()}: ${price_mean:.2f}\n"
        stats_text += f"Median {data_type.lower()}: ${price_median:.2f}\n"
        stats_text += f"Std deviation: ${price_std:.2f}\n"
        stats_text += f"Min {data_type.lower()}: ${price_min:.2f}\n"
        stats_text += f"Max {data_type.lower()}: ${price_max:.2f}\n"
        stats_text += f"Price range: ${price_max - price_min:.2f}"
        
        # Add text to the figure
        self.historical_fig.text(0.5, 0.01, stats_text, ha='center', va='bottom', fontsize=10, 
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Tight layout
        self.historical_fig.tight_layout(rect=[0, 0.1, 1, 1])
        
    def _plot_financial_statement_data(self, ticker, df, column_mapping):
        """Plot financial statement data in long format"""
        # Filter data for the selected ticker
        ticker_data = df[df['ticker'] == ticker].copy()
        
        if ticker_data.empty:
            # Show message if no data for this ticker
            ax = self.historical_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"No financial data available for {ticker}.", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.set_axis_off()
            return
            
        # Get unique dates and sort them
        dates = sorted(ticker_data['date'].unique())
        
        # If there are too many dates, use the most recent ones
        if len(dates) > 10:
            dates = dates[-10:]  # Last 10 dates
            ticker_data = ticker_data[ticker_data['date'].isin(dates)]
        
        # Get unique metrics and sort them
        metrics = sorted(ticker_data['metric'].unique())
        
        # If there are too many metrics, focus on the important ones
        important_metrics = [
            'revenue', 'net_income', 'total_assets', 'total_liabilities',
            'cash', 'debt', 'equity', 'earnings_per_share', 'dividend',
            'operating_income', 'gross_profit'
        ]
        
        # Filter for important metrics if we have too many
        if len(metrics) > 15:
            # Keep only important metrics that are in our data
            filtered_metrics = [m for m in important_metrics if m in metrics]
            
            # If we found some, use those, otherwise use the first 10
            if filtered_metrics:
                metrics = filtered_metrics
            else:
                metrics = metrics[:10]
                
            ticker_data = ticker_data[ticker_data['metric'].isin(metrics)]
        
        # Check if we still have data after filtering
        if ticker_data.empty:
            ax = self.historical_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"No relevant financial data found for {ticker}.", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.set_axis_off()
            return
        
        # Create a pivot table for easier plotting
        try:
            pivot_data = ticker_data.pivot_table(
                index='date', 
                columns='metric', 
                values='value',
                aggfunc='first'  # In case of duplicates
            )
            
            # Reset index to make date a column
            pivot_data = pivot_data.reset_index()
        except Exception as e:
            print(f"Error creating pivot table: {e}")
            ax = self.historical_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"Error processing data: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.set_axis_off()
            return
            
        # Create a figure with multiple subplots for important metrics
        n_metrics = len(pivot_data.columns) - 1  # Subtract 1 for date column
        
        if n_metrics == 0:
            ax = self.historical_fig.add_subplot(111)
            ax.text(0.5, 0.5, f"No metrics available for {ticker}.", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, fontsize=12)
            ax.set_axis_off()
            return
            
        # Determine grid layout based on number of metrics
        if n_metrics <= 4:
            rows, cols = 2, 2
        elif n_metrics <= 6:
            rows, cols = 2, 3
        elif n_metrics <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 4, 3  # Maximum 12 metrics
            
        # Create subplots
        for i, metric in enumerate(pivot_data.columns[1:]):  # Skip date column
            if i >= rows * cols:
                break  # Don't create more subplots than our grid allows
                
            ax = self.historical_fig.add_subplot(rows, cols, i+1)
            
            # Plot the metric
            ax.bar(pivot_data['date'], pivot_data[metric], color='blue', alpha=0.7)
            
            # Format title and labels
            formatted_metric = metric.replace('_', ' ').title()
            ax.set_title(formatted_metric, fontsize=10)
            
            # Format y-axis with K, M, B suffixes for large numbers
            ax.yaxis.set_major_formatter(plt.FuncFormatter(self._format_currency))
            
            # Rotate date labels if we have more than 3 dates
            if len(pivot_data['date']) > 3:
                plt.setp(ax.get_xticklabels(), rotation=45, fontsize=8)
            
            # Set tight layout within each subplot
            ax.grid(True, linestyle='--', alpha=0.6)
        
        # Add title
        self.historical_fig.suptitle(f"{ticker} - Financial Metrics", fontsize=14)
        
        # Adjust layout
        self.historical_fig.tight_layout()
        self.historical_fig.subplots_adjust(top=0.9)  # Make room for the suptitle
    
    def _format_currency(self, x, pos):
        """Format large numbers with K, M, B suffixes"""
        if x >= 1e9:
            return f'${x*1e-9:.1f}B'
        elif x >= 1e6:
            return f'${x*1e-6:.1f}M'
        elif x >= 1e3:
            return f'${x*1e-3:.1f}K'
        else:
            return f'${x:.1f}'
        
    def show_training_results(self, results):
        """Show training results"""
        # Store the results
        self.training_results = results
        
        # Update the ticker dropdown
        tickers = list(results.keys())
        self.training_ticker_combo['values'] = tickers
        
        if tickers:
            self.training_ticker_combo.current(0)
            self._on_training_ticker_selected(None)
            
        # Update prediction tab ticker dropdown
        self.prediction_ticker_combo['values'] = tickers
        if tickers:
            self.prediction_ticker_combo.current(0)
        
        # Switch to the training tab
        self.notebook.select(1)
        
    def _on_training_ticker_selected(self, event):
        """Handle training ticker selection"""
        ticker = self.training_ticker_var.get()
        if ticker in self.training_results:
            result = self.training_results[ticker]
            
            # Plot training history
            self.training_fig.clear()
            
            # Get the history object
            history = result['history']
            
            # Create subplot for loss
            ax1 = self.training_fig.add_subplot(211)
            ax1.plot(history.history['loss'], label='Training Loss')
            ax1.plot(history.history['val_loss'], label='Validation Loss')
            ax1.set_title(f'{ticker} - Training Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss (MSE)')
            ax1.grid(True)
            ax1.legend()
            
            # Create subplot for model summary
            ax2 = self.training_fig.add_subplot(212)
            
            # Model performance metrics
            performance = result['performance']
            
            # Calculate epochs run (accounting for early stopping)
            epochs_run = len(history.history['loss'])
            
            # Get model parameters
            model = result['model']
            
            # Create a model summary as text
            summary_text = f"Model: {model.__class__.__name__}\n"
            summary_text += f"Layers: {len(model.layers)} layers\n"
            summary_text += f"Test Loss (MSE): {performance:.6f}\n"
            summary_text += f"Epochs: {epochs_run}\n"
            
            # Add parameter count
            total_params = model.count_params()
            summary_text += f"Total Parameters: {total_params:,}\n"
            
            # Display the summary text
            ax2.text(0.5, 0.5, summary_text, 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax2.transAxes, fontsize=10)
            ax2.set_axis_off()
            
            # Tight layout
            self.training_fig.tight_layout()
            
            # Draw canvas
            self.training_canvas.draw()
            
    def _on_prediction_ticker_selected(self, event):
        """Handle prediction ticker selection"""
        # This doesn't plot anything yet, just prepares for prediction
        pass
        
    def _on_predict_clicked(self):
        """Handle predict button click"""
        ticker = self.prediction_ticker_var.get()
        days = self.prediction_days_var.get()
        
        if not ticker:
            return
            
        if ticker in self.training_results:
            # Import necessary modules here to avoid circular imports
            from models.lstm_model import predict_future_prices
            
            # Get the trained model and scaler
            model = self.training_results[ticker]['model']
            scaler = self.training_results[ticker]['scaler']
            
            try:
                # Find the main application window to call get_last_sequence
                # Start with the immediate parent and traverse up until we find the main app
                main_app = self._find_main_app()
                
                if main_app:
                    # Get the last sequence from the main app
                    last_sequence = main_app.get_last_sequence(ticker)
                    
                    # Predict future prices
                    predictions = predict_future_prices(model, last_sequence, scaler, days_to_predict=days)
                    
                    # Plot predictions
                    self._plot_predictions(ticker, predictions)
                    
                    # Store predictions
                    self.prediction_results[ticker] = predictions
                    
                    # Switch to prediction tab
                    self.notebook.select(2)
                else:
                    raise ValueError("Could not find main application window")
            except Exception as e:
                print(f"Error making predictions: {str(e)}")
                messagebox.showerror("Prediction Error", f"Failed to generate predictions: {str(e)}")
                
    def _find_main_app(self):
        """Find the main application instance (StockAnalyzerApp) by traversing up the widget hierarchy"""
        # Start with the immediate parent
        parent = self.parent
        
        # Keep going up until we find the main app or reach the top
        while parent:
            # Check if this parent has the get_last_sequence method
            if hasattr(parent, 'get_last_sequence'):
                return parent
                
            # Try to get the parent's parent
            try:
                parent = parent.master
            except:
                # Reached the top without finding the main app
                return None
        
    def _plot_predictions(self, ticker, predictions):
        """Plot price predictions with normalization details"""
        # Clear previous plot
        self.prediction_fig.clear()
        
        # Create subplot for price predictions
        ax1 = self.prediction_fig.add_subplot(211)
        
        # Create date range for x-axis
        import datetime
        today = datetime.datetime.now()
        dates = [today + datetime.timedelta(days=i) for i in range(len(predictions))]
        
        # Calculate prediction uncertainty (simple approximation)
        # Increase uncertainty as we predict further into the future
        uncertainty = np.linspace(0.01, 0.10, len(predictions)) * np.mean(predictions)
        lower_bound = predictions - uncertainty
        upper_bound = predictions + uncertainty
        
        # Plot predicted prices
        ax1.plot(dates, predictions, label='Predicted Price', color='red', linewidth=2)
        
        # Add confidence interval
        ax1.fill_between(dates, lower_bound, upper_bound, alpha=0.2, color='red',
                          label='Prediction Uncertainty')
        
        # Add title and labels
        ax1.set_title(f'{ticker} - Price Predictions for Next {len(predictions)} Days')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Predicted Price')
        ax1.grid(True)
        ax1.legend()
        
        # Rotate date labels
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # Create subplot for normalization info and prediction details
        ax2 = self.prediction_fig.add_subplot(212)
        
        # Get prediction details from the training results
        result = self.training_results[ticker]
        model = result['model']
        scaler = result['scaler']
        
        # Create a summary of the prediction details
        details_text = f"Prediction Details for {ticker}\n\n"
        details_text += f"Model: {model.__class__.__name__}\n"
        details_text += f"Forecast Horizon: {len(predictions)} days\n"
        
        # Normalization details
        details_text += "\nNormalization Information:\n"
        
        if hasattr(scaler, 'feature_range'):
            details_text += f"Normalization Type: MinMaxScaler (range {scaler.feature_range})\n"
        else:
            details_text += f"Normalization Type: StandardScaler\n"
            
        # Price range details
        details_text += f"Predicted Price Range: ${min(predictions):.2f} - ${max(predictions):.2f}\n"
        
        # Confidence info
        details_text += f"Confidence Range: Initial ±{uncertainty[0]:.2f}, Final ±{uncertainty[-1]:.2f}\n"
        
        # Add note about normalization process
        details_text += "\nNote: Predictions are made using normalized data and then\n"
        details_text += "converted back to original price scale for display."
        
        # Display prediction details
        ax2.text(0.5, 0.5, details_text,
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=10, linespacing=1.5)
        ax2.set_axis_off()
        
        # Tight layout
        self.prediction_fig.tight_layout()
        
        # Draw canvas
        self.prediction_canvas.draw()
        
    def _plot_sentiment_data(self, ticker, df, column_mapping):
        """Plot sentiment data for a ticker"""
        # Sort data by date
        df = df.sort_values(by=column_mapping['date'])
        
        # Create a subplot for sentiment
        ax1 = self.historical_fig.add_subplot(111)
        
        date_col = column_mapping['date']
        
        # If we have a sentiment score column, plot it
        if 'sentiment' in column_mapping:
            sentiment_col = column_mapping['sentiment']
            
            # Create a colormap based on sentiment values
            colors = []
            for sentiment in df[sentiment_col]:
                if sentiment > 0:
                    colors.append('green')
                elif sentiment < 0:
                    colors.append('red')
                else:
                    colors.append('gray')
            
            # Plot sentiment as a bar chart
            bars = ax1.bar(df[date_col], df[sentiment_col], alpha=0.7, color=colors, label='Sentiment')
            
            # Add a horizontal line at y=0 to show positive/negative boundary
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Add title and labels
            ax1.set_title(f'{ticker} - Sentiment Analysis')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Sentiment Score')
            ax1.grid(True, alpha=0.3)
            
            # Rotate date labels
            plt.setp(ax1.get_xticklabels(), rotation=45)
            
            # Add average sentiment as text
            avg_sentiment = df[sentiment_col].mean()
            sentiment_status = "Positive" if avg_sentiment > 0 else "Negative" if avg_sentiment < 0 else "Neutral"
            text_color = "green" if avg_sentiment > 0 else "red" if avg_sentiment < 0 else "gray"
            
            ax1.text(0.02, 0.95, f"Average Sentiment: {avg_sentiment:.2f} ({sentiment_status})", 
                    transform=ax1.transAxes, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.8),
                    color=text_color)
            
            # If we have detailed sentiment components, add a second subplot
            if any(metric in column_mapping for metric in ['positive', 'negative', 'neutral']):
                # Adjust the first subplot
                ax1.set_position([0.125, 0.5, 0.775, 0.4])
                
                # Create a second subplot for detailed sentiment
                ax2 = self.historical_fig.add_subplot(212)
                
                # Stack bars for positive, negative, neutral if available
                bottom = np.zeros(len(df))
                for sentiment_type, color in [('positive', 'green'), ('neutral', 'gray'), ('negative', 'red')]:
                    if sentiment_type in column_mapping:
                        ax2.bar(df[date_col], df[column_mapping[sentiment_type]], 
                                bottom=bottom, label=sentiment_type.capitalize(), 
                                alpha=0.7, color=color)
                        bottom += df[column_mapping[sentiment_type]].values
                
                # Add title and labels
                ax2.set_title(f'{ticker} - Detailed Sentiment Components')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Component Strength')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                # Rotate date labels
                plt.setp(ax2.get_xticklabels(), rotation=45)
            
            # Tight layout
            self.historical_fig.tight_layout()
        
    def load_tickers_for_historical(self, tickers, plot_first=True):
        """
        Load tickers into the historical tab dropdown and optionally plot the first one
        
        Args:
            tickers: List of ticker symbols to add to the dropdown
            plot_first: Whether to automatically plot the first ticker
        """
        if not tickers:
            return
            
        print(f"Loading tickers for historical tab: {tickers}")
        # Update the ticker dropdown
        self.historical_ticker_combo['values'] = tickers
        
        # Select the first ticker
        self.historical_ticker_var.set(tickers[0])
        
        # If requested and we have historical data for the first ticker, plot it
        if plot_first and tickers[0] in self.historical_data:
            print(f"Auto-plotting first ticker: {tickers[0]}")
            self._plot_historical_data(tickers[0])
        
    def show_historical_data(self, df, ticker):
        """
        Store and display historical data for the given ticker
        
        Args:
            df: DataFrame containing the historical data
            ticker: Ticker symbol
        """
        # Store the data in our historical_data dictionary
        self.historical_data[ticker] = df
        
        # If this is the ticker currently selected in the dropdown, plot it
        current_ticker = self.historical_ticker_var.get()
        if current_ticker == ticker:
            self._plot_historical_data(ticker)
        elif not current_ticker and ticker:
            # If no ticker is currently selected, select this one
            self.historical_ticker_var.set(ticker)
            # This will trigger the _on_historical_ticker_selected event
        
        # Switch to the historical tab if not already there
        if self.notebook.index("current") != 0:  # 0 is the index of the historical tab
            self.notebook.select(0)  # Select the historical tab
        
        # Update the historical tab with the data
        print(f"Historical data loaded for {ticker}")
        
    def initialize_prediction_tab(self):
        """Initialize the prediction tab with a new method name"""
        # Create a frame for controls
        self.prediction_control_frame = ttk.Frame(self.prediction_tab)
        self.prediction_control_frame.pack(fill="x", padx=5, pady=5)
        
        # Ticker selection for predictions
        ttk.Label(self.prediction_control_frame, text="Ticker:").pack(side="left", padx=5)
        self.prediction_ticker_var = tk.StringVar()
        self.prediction_ticker_combo = ttk.Combobox(self.prediction_control_frame, 
                                                   textvariable=self.prediction_ticker_var,
                                                   state="readonly", width=10)
        self.prediction_ticker_combo.pack(side="left", padx=5)
        self.prediction_ticker_combo.bind("<<ComboboxSelected>>", self._on_prediction_ticker_selected)
        
        # Days to predict
        ttk.Label(self.prediction_control_frame, text="Days:").pack(side="left", padx=5)
        self.prediction_days_var = tk.IntVar(value=30)
        days_spin = ttk.Spinbox(self.prediction_control_frame, from_=1, to=365, 
                               textvariable=self.prediction_days_var, width=5)
        days_spin.pack(side="left", padx=5)
        
        # Predict button
        self.predict_btn = ttk.Button(self.prediction_control_frame, text="Predict", 
                                     command=self._on_predict_clicked)
        self.predict_btn.pack(side="left", padx=10)
        
        # Create a frame for the matplotlib figure
        self.prediction_figure_frame = ttk.Frame(self.prediction_tab)
        self.prediction_figure_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Create figure and canvas
        self.prediction_fig = Figure(figsize=(10, 6), dpi=100)
        self.prediction_canvas = FigureCanvasTkAgg(self.prediction_fig, self.prediction_figure_frame)
        self.prediction_canvas.get_tk_widget().pack(fill="both", expand=True)
        
        # Add toolbar
        self.prediction_toolbar = NavigationToolbar2Tk(self.prediction_canvas, self.prediction_figure_frame)
        self.prediction_toolbar.update()
        
        # Initial message
        ax = self.prediction_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Train a model and make predictions", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_axis_off()
        self.prediction_canvas.draw()
        
        # Store prediction results
        self.prediction_results = {} 