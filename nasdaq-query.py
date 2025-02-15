import duckdb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import atexit

class StockDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Market Dashboard")
        
        # Connect to database
        self.db_name = "nasdaq_stocks.db"
        self.con = duckdb.connect(self.db_name)
        
        # Register cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Create control frame
        self.control_frame = ttk.Frame(root)
        self.control_frame.pack(fill='x', padx=5, pady=5)
        
        # Get available tickers
        self.tickers = self.con.execute("SELECT DISTINCT ticker FROM stock_prices ORDER BY ticker").df()['ticker'].tolist()
        
        # Create ticker selection
        ttk.Label(self.control_frame, text="Select Ticker:").pack(side='left', padx=5)
        self.ticker_var = tk.StringVar(value='AAPL')
        self.ticker_combo = ttk.Combobox(
            self.control_frame, 
            textvariable=self.ticker_var,
            values=self.tickers,
            state='readonly'
        )
        self.ticker_combo.pack(side='left', padx=5)
        
        # Create update button
        self.update_btn = ttk.Button(
            self.control_frame,
            text="Update Charts",
            command=self.update_plots
        )
        self.update_btn.pack(side='left', padx=5)
        
        # Create matplotlib figure
        self.fig = plt.figure(figsize=(15, 12))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Initial plot
        self.update_plots()
    
    def on_closing(self):
        """Cleanup when window is closed"""
        if hasattr(self, 'con') and self.con:
            try:
                self.con.close()
            except:
                pass
        self.root.destroy()
        
    def update_plots(self):
        try:
            selected_ticker = self.ticker_var.get()
            self.fig.clear()
            plt.style.use('ggplot')
            self.fig.suptitle('Stock Market Analysis Dashboard', fontsize=16)
            
            # Get data
            price_data = self.con.execute("""
                SELECT date, ticker, close, volume
                FROM stock_prices
                WHERE date >= (SELECT MAX(date) - INTERVAL '30 days' FROM stock_prices)
                    AND ticker = ?
                ORDER BY date
            """, [selected_ticker]).df()
            
            tech_data = self.con.execute("""
                WITH moving_avg AS (
                    SELECT 
                        date,
                        close,
                        AVG(close) OVER (ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as ma_20
                    FROM stock_prices
                    WHERE ticker = ?
                    ORDER BY date
                )
                SELECT * FROM moving_avg
                WHERE ma_20 IS NOT NULL
            """, [selected_ticker]).df()
            
            signals_data = self.con.execute("""
                SELECT signal_type, COUNT(*) as count
                FROM trading_signals
                GROUP BY signal_type
            """).df()
            
            sector_data = self.con.execute("""
                SELECT 
                    s.sector,
                    COUNT(DISTINCT s.symbol) as company_count
                FROM stocks s
                GROUP BY s.sector
                ORDER BY company_count DESC
            """).df()
            
            # 1. Price Trends
            ax1 = self.fig.add_subplot(221)
            ax1.plot(price_data['date'], price_data['close'], label=selected_ticker)
            ax1.set_title(f'30-Day Price Trend ({selected_ticker})')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Close Price ($)')
            ax1.legend()
            ax1.tick_params(axis='x', rotation=45)
            
            # 2. Technical Analysis
            ax2 = self.fig.add_subplot(222)
            ax2.plot(tech_data['date'], tech_data['close'], label=f'{selected_ticker} Price')
            ax2.plot(tech_data['date'], tech_data['ma_20'], label='20-day MA', linestyle='--')
            ax2.set_title(f'Technical Analysis ({selected_ticker})')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price ($)')
            ax2.legend()
            ax2.tick_params(axis='x', rotation=45)
            
            # 3. Trading Signals
            ax3 = self.fig.add_subplot(223)
            ax3.bar(signals_data['signal_type'], signals_data['count'])
            ax3.set_title('Distribution of Trading Signals')
            ax3.set_xlabel('Signal Type')
            ax3.set_ylabel('Count')
            ax3.tick_params(axis='x', rotation=45)
            
            # 4. Sector Analysis
            ax4 = self.fig.add_subplot(224)
            ax4.pie(sector_data['company_count'], labels=sector_data['sector'], autopct='%1.1f%%')
            ax4.set_title('Companies by Sector')
            
            plt.tight_layout()
            self.canvas.draw()
        except Exception as e:
            print(f"Error updating plots: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = StockDashboard(root)
    root.mainloop()

