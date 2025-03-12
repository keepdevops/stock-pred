"""
Panel for displaying historical data
"""
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ui.visualization.base_panel import BasePanel

class HistoricalPanel(BasePanel):
    def __init__(self, parent, event_bus):
        super().__init__(parent, event_bus)
        
        # Create UI components
        self._create_ui()
        
        # Subscribe to events
        event_bus.subscribe("historical_data_updated", self._on_data_updated)
        event_bus.subscribe("historical_data_error", self._on_data_error)
        
    def _create_ui(self):
        """Create the UI components"""
        # Create a frame for controls
        controls_frame = ttk.Frame(self)
        controls_frame.pack(fill="x", padx=10, pady=5)
        
        # Create ticker selection dropdown
        ttk.Label(controls_frame, text="Ticker:").pack(side="left", padx=5)
        self.ticker_var = tk.StringVar()
        self.ticker_combo = ttk.Combobox(controls_frame, textvariable=self.ticker_var, state="readonly")
        self.ticker_combo.pack(side="left", padx=5)
        
        # Bind ticker selection event
        self.ticker_combo.bind("<<ComboboxSelected>>", self._on_ticker_selected)
        
        # Create view selection dropdown
        ttk.Label(controls_frame, text="View:").pack(side="left", padx=5)
        self.view_var = tk.StringVar()
        self.view_combo = ttk.Combobox(
            controls_frame, 
            textvariable=self.view_var,
            values=["Price", "Volume", "Price & Volume", "Statistics", 
                    "Financial Statements", "Sentiment",
                    "Candlestick", "Moving Averages", "Bollinger Bands", "RSI"],
            state="readonly"
        )
        self.view_combo.pack(side="left", padx=5)
        
        # Bind view selection event
        self.view_combo.bind("<<ComboboxSelected>>", self._on_view_selected)
        
        # Create radio buttons for view modes
        view_mode_frame = ttk.Frame(controls_frame)
        view_mode_frame.pack(side="right", padx=10)
        
        self.view_mode_var = tk.StringVar(value="Table")
        ttk.Radiobutton(view_mode_frame, text="Table", variable=self.view_mode_var, 
                       value="Table", command=self._on_view_mode_changed).pack(side="left")
        ttk.Radiobutton(view_mode_frame, text="Plot", variable=self.view_mode_var, 
                       value="Plot", command=self._on_view_mode_changed).pack(side="left")
        
        # Create a frame for the table
        self.table_frame = ttk.Frame(self)
        self.table_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Create a frame for the plot
        self.plot_frame = ttk.Frame(self)
        self.canvas.get_tk_widget().pack(in_=self.plot_frame, fill="both", expand=True)
        
        # Default view mode is table
        self.table_frame.pack(fill="both", expand=True)
        self.plot_frame.pack_forget()
        
    def set_tickers(self, tickers):
        """Set the available tickers"""
        self.ticker_combo['values'] = tickers
        if tickers:
            self.ticker_combo.current(0)
            self.event_bus.publish("historical_ticker_selected", {'ticker': tickers[0]})
            
    def _on_ticker_selected(self, event):
        """Handle ticker selection event"""
        ticker = self.ticker_var.get()
        if ticker:
            self.event_bus.publish("historical_ticker_selected", {'ticker': ticker})
            
    def _on_view_selected(self, event):
        """Handle view selection event"""
        ticker = self.ticker_var.get()
        view = self.view_var.get()
        if ticker and view:
            self.event_bus.publish("historical_view_selected", {
                'ticker': ticker,
                'view': view
            })
            
    def _on_view_mode_changed(self):
        """Handle view mode change"""
        mode = self.view_mode_var.get()
        
        if mode == "Table":
            self.plot_frame.pack_forget()
            self.table_frame.pack(fill="both", expand=True)
        else:
            self.table_frame.pack_forget()
            self.plot_frame.pack(fill="both", expand=True)
            
            # Update the plot if ticker and view are selected
            ticker = self.ticker_var.get()
            view = self.view_var.get()
            if ticker and view:
                self.event_bus.publish("historical_view_selected", {
                    'ticker': ticker,
                    'view': view
                })
                
    def _on_data_updated(self, data):
        """Handle historical data updated event"""
        ticker = data['ticker']
        historical_data = data['data']
        
        # Update table view
        self._update_table(historical_data)
        
        # Update plot if in plot mode
        if self.view_mode_var.get() == "Plot":
            view = self.view_var.get()
            self._plot_data(historical_data, view)
            
    def _on_data_error(self, data):
        """Handle historical data error event"""
        message = data['message']
        
        # Clear table
        for widget in self.table_frame.winfo_children():
            widget.destroy()
            
        # Show error message in table
        ttk.Label(self.table_frame, text=message).pack(pady=20)
        
        # Show error in plot
        self.show_error(message)
        
    def _update_table(self, historical_data):
        """Update the table with historical data"""
        # Clear existing table
        for widget in self.table_frame.winfo_children():
            widget.destroy()
            
        # Get dataframe
        df = historical_data.dataframe
        
        if df.empty:
            ttk.Label(self.table_frame, text=f"No data to display for {historical_data.ticker}").pack(pady=20)
            return
            
        # Create a new frame for the table
        table_container = ttk.Frame(self.table_frame)
        table_container.pack(fill="both", expand=True)
        
        # Create scrollbars
        x_scrollbar = ttk.Scrollbar(table_container, orient="horizontal")
        y_scrollbar = ttk.Scrollbar(table_container, orient="vertical")
        
        # Create the treeview widget
        columns = list(df.columns)
        tree = ttk.Treeview(table_container, columns=columns, show="headings",
                           xscrollcommand=x_scrollbar.set,
                           yscrollcommand=y_scrollbar.set)
        
        # Configure scrollbars
        x_scrollbar.config(command=tree.xview)
        y_scrollbar.config(command=tree.yview)
        
        # Pack scrollbars
        x_scrollbar.pack(side="bottom", fill="x")
        y_scrollbar.pack(side="right", fill="y")
        
        # Pack the treeview
        tree.pack(side="left", fill="both", expand=True)
        
        # Configure columns and headings
        for col in columns:
            tree.column(col, width=100, anchor="center")
            tree.heading(col, text=col)
            
        # Insert data rows
        for i, row in df.iterrows():
            # Format the values
            values = [self._format_value(val) for val in row]
            tree.insert("", "end", values=values)
            
    def _format_value(self, value):
        """Format a value for display"""
        if isinstance(value, (int, float)):
            if abs(value) > 1000000000:
                return f"{value/1000000000:.2f}B"
            elif abs(value) > 1000000:
                return f"{value/1000000:.2f}M"
            elif abs(value) > 1000:
                return f"{value/1000:.2f}K"
            else:
                return f"{value:.2f}"
        else:
            return str(value)
            
    def _plot_data(self, historical_data, view):
        """Plot historical data"""
        # Clear existing plot
        self.clear_plot()
        
        # Get dataframe and column mapping
        df = historical_data.dataframe
        column_mapping = historical_data.column_mapping
        ticker = historical_data.ticker
        
        if df.empty:
            self.show_error(f"No data to plot for {ticker}")
            return
            
        try:
            # Plot based on view type
            if view == "Price":
                self._plot_price_data(ticker, df, column_mapping)
            elif view == "Volume":
                self._plot_volume_data(ticker, df, column_mapping)
            elif view == "Price & Volume":
                self._plot_price_and_volume_data(ticker, df, column_mapping)
            elif view == "Statistics":
                self._plot_statistics_data(ticker, df, column_mapping)
            elif view == "Financial Statements":
                self._plot_financial_statement_data(ticker, df, column_mapping)
            elif view == "Sentiment":
                self._plot_sentiment_data(ticker, df, column_mapping)
            elif view == "Candlestick":
                self._plot_candlestick_data(ticker, df, column_mapping)
            elif view == "Moving Averages":
                self._plot_moving_averages(ticker, df, column_mapping)
            elif view == "Bollinger Bands":
                self._plot_bollinger_bands(ticker, df, column_mapping)
            elif view == "RSI":
                self._plot_rsi(ticker, df, column_mapping)
            else:
                self.show_error(f"Unknown view type: {view}")
                
            # Update the canvas
            self.canvas.draw()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.show_error(f"Error plotting data: {str(e)}")
            
    # Various plotting methods would be defined here...
    # For brevity, I'll include just one example method:
    
    def _plot_price_data(self, ticker, df, column_mapping):
        """Plot price data"""
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        # Check if we have date and price data
        if 'Date' in column_mapping and 'Close' in column_mapping:
            date_col = column_mapping['Date']
            close_col = column_mapping['Close']
            
            # Convert dates to datetime
            dates = pd.to_datetime(df[date_col])
            
            # Plot close price
            ax.plot(dates, df[close_col], label='Close Price')
            
            # Add other price data if available
            if 'Open' in column_mapping:
                open_col = column_mapping['Open']
                ax.plot(dates, df[open_col], label='Open Price')
                
            if 'High' in column_mapping:
                high_col = column_mapping['High']
                ax.plot(dates, df[high_col], label='High Price')
                
            if 'Low' in column_mapping:
                low_col = column_mapping['Low']
                ax.plot(dates, df[low_col], label='Low Price')
            
            ax.set_title(f"{ticker} Price Data")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True)
            
            # Format date axis
            plt.xticks(rotation=45)
            plt.tight_layout()
        else:
            self.show_error("Price data not available") 