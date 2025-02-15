import duckdb
import matplotlib
matplotlib.use('TkAgg')  # Set backend before any other matplotlib imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk, colorchooser
import sys
import atexit
import pandas as pd
import numpy as np
import mplfinance as mpf

# Global cleanup function
def cleanup():
    plt.close('all')

# Register cleanup
atexit.register(cleanup)

def initialize_database(db_path="forex-duckdb.db"):
    """Initialize the database with required tables"""
    try:
        con = duckdb.connect(db_path)
        
        # Create forex_prices table
        con.execute("""
            CREATE TABLE IF NOT EXISTS forex_prices (
                id INTEGER PRIMARY KEY,
                pair VARCHAR NOT NULL,
                date TIMESTAMP NOT NULL,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create index on pair and date
        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_forex_pair_date 
            ON forex_prices(pair, date)
        """)
        
        print("Database tables initialized successfully")
        return True
        
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        return False
    finally:
        if 'con' in locals():
            con.close()

def plot_forex_data(pair, db_path="forex-duckdb.db"):
    """Plot forex data for a specific pair"""
    import seaborn as sns
    
    try:
        # Connect to DuckDB
        con = duckdb.connect(db_path)
        
        # Query the data
        query = f"""
            SELECT date, open, high, low, close, volume
            FROM forex_prices
            WHERE pair = '{pair}'
            ORDER BY date
        """
        
        # Execute query and convert to pandas
        df = con.execute(query).df()
        
        if len(df) == 0:
            print(f"No data found for {pair}")
            return
            
        # Create figure and axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        fig.suptitle(f'{pair} Price and Volume')
        
        # Plot price data
        ax1.plot(df['date'], df['close'], label='Close Price')
        ax1.set_title('Price History')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        
        # Plot volume
        ax2.bar(df['date'], df['volume'], label='Volume')
        ax2.set_title('Volume')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting data: {str(e)}")
    finally:
        if 'con' in locals():
            con.close()

def plot_forex_candlestick(pair, db_path="forex-duckdb.db"):
    """Plot candlestick chart for a forex pair with customizable colors"""
    import mplfinance as mpf
    import numpy as np
    
    try:
        # Connect to DuckDB
        con = duckdb.connect(db_path)
        
        # Query the data
        query = f"""
            SELECT date, open, high, low, close, 
                   CASE WHEN volume = 0 OR volume IS NULL 
                        THEN 0.000001 
                        ELSE volume 
                   END as volume
            FROM forex_prices
            WHERE pair = '{pair}'
            ORDER BY date
        """
        df = con.execute(query).df()
        
        if len(df) == 0:
            print(f"No data found for {pair}")
            return
            
        # Convert date to datetime and set as index
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Ensure volume values are valid for plotting
        df['volume'] = df['volume'].replace({0: np.nan})
        df['volume'] = df['volume'].fillna(df['volume'].mean() * 0.001)
        
        # Create window
        plot_window = tk.Toplevel()
        plot_window.title(f"{pair} Candlestick Chart")
        plot_window.geometry("1200x800")
        
        # Default colors
        default_colors = {
            'candle_up': '#26a69a',    # Green
            'candle_down': '#ef5350',   # Red
            'edge_up': '#00796b',       # Dark Green
            'edge_down': '#c62828',     # Dark Red
            'wick_up': '#004d40',       # Darker Green
            'wick_down': '#b71c1c',     # Darker Red
            'volume_up': '#26a69a',     # Green
            'volume_down': '#ef5350'    # Red
        }
        
        # Color variables
        color_vars = {k: tk.StringVar(value=v) for k, v in default_colors.items()}
        
        def pick_color(component):
            try:
                color = colorchooser.askcolor(color=color_vars[component].get(), 
                                            title=f"Choose {component.replace('_', ' ').title()} Color")[1]
                if color:
                    color_vars[component].set(color)
                    color_buttons[component].configure(bg=color)
                    redraw_plot()
            except Exception as e:
                print(f"Error picking color: {str(e)}")
        
        def redraw_plot():
            try:
                for widget in plot_frame.winfo_children():
                    widget.destroy()
                
                # Create marketcolors with current color settings
                mc = mpf.make_marketcolors(
                    up=color_vars['candle_up'].get(),
                    down=color_vars['candle_down'].get(),
                    edge={'up': color_vars['edge_up'].get(), 
                         'down': color_vars['edge_down'].get()},
                    wick={'up': color_vars['wick_up'].get(), 
                         'down': color_vars['wick_down'].get()},
                    volume={'up': color_vars['volume_up'].get(), 
                           'down': color_vars['volume_down'].get()},
                    inherit=False)
                
                # Create style
                s = mpf.make_mpf_style(
                    marketcolors=mc,
                    gridstyle='dotted',
                    y_on_right=False)
                
                # Create the plot
                fig, axlist = mpf.plot(
                    df, 
                    type='candle',
                    volume=True,
                    style=s,
                    figsize=(12, 8),
                    panel_ratios=(3, 1),
                    title=f'\n{pair} Price and Volume',
                    returnfig=True
                )
                
                # Create canvas
                canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                canvas.draw()
                
                # Add toolbar
                toolbar = NavigationToolbar2Tk(canvas, plot_frame)
                toolbar.update()
                
                # Pack widgets
                toolbar.pack(side=tk.TOP, fill=tk.X)
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            
            except Exception as e:
                print(f"Error redrawing plot: {str(e)}")
        
        def on_closing():
            try:
                plt.close('all')
                plot_window.destroy()
            except Exception as e:
                print(f"Error closing window: {str(e)}")
        
        plot_window.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Create control frames
        control_frame1 = ttk.Frame(plot_window, padding="5")
        control_frame1.pack(side=tk.TOP, fill=tk.X)
        
        control_frame2 = ttk.Frame(plot_window, padding="5")
        control_frame2.pack(side=tk.TOP, fill=tk.X)
        
        # Color buttons
        color_buttons = {}
        
        # First row: Candle and Edge colors
        components1 = ['candle_up', 'candle_down', 'edge_up', 'edge_down']
        for component in components1:
            ttk.Label(control_frame1, 
                     text=f"{component.replace('_', ' ').title()}:").pack(side=tk.LEFT, padx=5)
            color_buttons[component] = tk.Button(
                control_frame1,
                text="Pick Color",
                command=lambda c=component: pick_color(c),
                bg=color_vars[component].get(),
                width=10
            )
            color_buttons[component].pack(side=tk.LEFT, padx=5)
        
        # Second row: Wick and Volume colors
        components2 = ['wick_up', 'wick_down', 'volume_up', 'volume_down']
        for component in components2:
            ttk.Label(control_frame2, 
                     text=f"{component.replace('_', ' ').title()}:").pack(side=tk.LEFT, padx=5)
            color_buttons[component] = tk.Button(
                control_frame2,
                text="Pick Color",
                command=lambda c=component: pick_color(c),
                bg=color_vars[component].get(),
                width=10
            )
            color_buttons[component].pack(side=tk.LEFT, padx=5)
        
        # Create plot frame
        plot_frame = ttk.Frame(plot_window)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Initial plot
        redraw_plot()
        
    except Exception as e:
        print(f"Error plotting candlestick: {str(e)}")
    finally:
        if 'con' in locals():
            con.close()

def get_forex_summary(db_path="forex-duckdb.db"):
    """Get summary of available forex data"""
    try:
        con = duckdb.connect(db_path)
        
        # Get summary statistics
        query = """
            SELECT 
                pair,
                MIN(date) as start_date,
                MAX(date) as end_date,
                COUNT(*) as days,
                ROUND(AVG(close), 4) as avg_price,
                ROUND(MIN(close), 4) as min_price,
                ROUND(MAX(close), 4) as max_price
            FROM forex_prices
            GROUP BY pair
            ORDER BY pair
        """
        
        results = con.execute(query).df()
        print("\nForex Data Summary:")
        print(results.to_string(index=False))
        
        return results
        
    except Exception as e:
        print(f"Error getting summary: {str(e)}")
    finally:
        if 'con' in locals():
            con.close()

def plot_multiple_forex_data(pairs, db_path="forex-duckdb.db"):
    """Plot forex data for multiple pairs on the same chart"""
    import seaborn as sns
    
    try:
        # Connect to DuckDB
        con = duckdb.connect(db_path)
        
        # Create figure and axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
        fig.suptitle('Multiple Forex Pairs Comparison')
        
        for pair in pairs:
            # Query the data
            query = f"""
                SELECT date, open, high, low, close, volume
                FROM forex_prices
                WHERE pair = '{pair}'
                ORDER BY date
            """
            
            # Execute query and convert to pandas
            df = con.execute(query).df()
            
            if len(df) == 0:
                print(f"No data found for {pair}")
                continue
                
            # Plot price data
            ax1.plot(df['date'], df['close'], label=pair)
            
            # Plot volume
            ax2.bar(df['date'], df['volume'], label=pair, alpha=0.5)
        
        # Customize plots
        ax1.set_title('Price History')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        ax1.grid(True)
        ax1.legend()
        
        ax2.set_title('Volume')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Volume')
        ax2.grid(True)
        ax2.legend()
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Error plotting data: {str(e)}")
    finally:
        if 'con' in locals():
            con.close()

def plot_forex_line(pair, db_path="forex-duckdb.db", colors=None):
    """Plot line chart for a forex pair with customizable colors"""
    try:
        # Connect to DuckDB
        con = duckdb.connect(db_path)
        
        # Query the data with volume handling
        query = f"""
            SELECT date, open, high, low, close, 
                   CASE WHEN volume = 0 OR volume IS NULL 
                        THEN 0.000001 
                        ELSE volume 
                   END as volume,
                   CASE WHEN close > open THEN 1 ELSE 0 END as price_up
            FROM forex_prices
            WHERE pair = '{pair}'
            ORDER BY date
        """
        df = con.execute(query).df()
        
        if len(df) == 0:
            print(f"No data found for {pair}")
            return
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Ensure volume values are valid for plotting
        df['volume'] = df['volume'].replace({0: np.nan})
        df['volume'] = df['volume'].fillna(df['volume'].mean() * 0.001)
        
        # Create window
        plot_window = tk.Toplevel()
        plot_window.title(f"{pair} Line Plot")
        plot_window.geometry("1200x800")
        
        # Default colors
        default_colors = {
            'open': '#2196F3',    # Blue
            'high': '#4CAF50',    # Green
            'low': '#F44336',     # Red
            'close': '#9C27B0',   # Purple
            'volume': '#90CAF9'   # Light Blue
        }
        colors = colors or default_colors
        
        def pick_color(price_type):
            try:
                color = colorchooser.askcolor(color=color_vars[price_type].get(), 
                                            title=f"Choose {price_type.title()} Color")[1]
                if color:
                    color_vars[price_type].set(color)
                    color_buttons[price_type].configure(bg=color)
                    redraw_plot()
            except Exception as e:
                print(f"Error picking color: {str(e)}")
        
        def redraw_plot():
            try:
                for widget in plot_frame.winfo_children():
                    widget.destroy()
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[3, 1])
                fig.suptitle(f'{pair} Price and Volume')
                
                # Plot OHLC data
                ax1.plot(df['date'], df['open'], label='Open', 
                        color=color_vars['open'].get(), alpha=0.7)
                ax1.plot(df['date'], df['high'], label='High', 
                        color=color_vars['high'].get(), alpha=0.7)
                ax1.plot(df['date'], df['low'], label='Low', 
                        color=color_vars['low'].get(), alpha=0.7)
                ax1.plot(df['date'], df['close'], label='Close', 
                        color=color_vars['close'].get(), linewidth=2)
                
                ax1.set_title('Price History')
                ax1.set_xlabel('Date')
                ax1.set_ylabel('Price')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Plot volume with color based on price movement
                volume_colors = np.where(df['price_up'] == 1, 
                                       color_vars['volume'].get(), 
                                       color_vars['volume'].get())
                ax2.bar(df['date'], df['volume'], label='Volume', 
                       color=volume_colors, alpha=0.7)
                ax2.set_title('Volume')
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Volume')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
                
                plt.tight_layout()
                
                canvas = FigureCanvasTkAgg(fig, master=plot_frame)
                canvas.draw()
                
                toolbar = NavigationToolbar2Tk(canvas, plot_frame)
                toolbar.update()
                
                toolbar.pack(side=tk.TOP, fill=tk.X)
                canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            except Exception as e:
                print(f"Error redrawing plot: {str(e)}")
        
        def on_closing():
            try:
                plt.close('all')
                plot_window.destroy()
            except Exception as e:
                print(f"Error closing window: {str(e)}")
        
        plot_window.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Create control frame
        control_frame = ttk.Frame(plot_window, padding="5")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Color variables and buttons
        color_vars = {}
        color_buttons = {}
        
        # Create color controls for each price type
        for price_type, color in colors.items():
            color_vars[price_type] = tk.StringVar(value=color)
            
            ttk.Label(control_frame, 
                     text=f"{price_type.title()} Color:").pack(side=tk.LEFT, padx=5)
            
            color_buttons[price_type] = tk.Button(
                control_frame, 
                text=f"Pick {price_type.title()}", 
                command=lambda pt=price_type: pick_color(pt),
                bg=color,
                width=10
            )
            color_buttons[price_type].pack(side=tk.LEFT, padx=5)
        
        # Create plot frame
        plot_frame = ttk.Frame(plot_window)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Initial plot
        redraw_plot()
        
    except Exception as e:
        print(f"Error plotting line chart: {str(e)}")
    finally:
        if 'con' in locals():
            con.close()

def create_gui():
    """Create a Tkinter GUI for forex pair selection and plotting"""
    from tkinter import ttk
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import matplotlib.pyplot as plt
    import sys
    
    root = tk.Tk()
    root.title("Forex Pair Plotter")
    root.geometry("400x500")
    
    def on_closing(window=None):
        """Handle window closing"""
        plt.close('all')  # Close all figures
        if window:
            window.destroy()
        else:
            root.quit()
            root.destroy()
            sys.exit(0)
    
    def handle_keypress(event):
        """Handle keyboard shortcuts"""
        if event.state == 4 and event.keysym == 'c':  # Ctrl+C
            on_closing()
    
    def plot_selected():
        selected_pairs = [pairs_listbox.get(i) for i in pairs_listbox.curselection()]
        if selected_pairs:
            if plot_type.get() == "Line Plot":
                for pair in selected_pairs:
                    plot_forex_line(pair)
            else:  # Candlestick Plot
                for pair in selected_pairs:
                    plot_forex_candlestick(pair)
    
    # Get available pairs from database
    summary = get_forex_summary()
    available_pairs = summary['pair'].tolist() if not summary.empty else []
    
    # Add keyboard binding
    root.bind('<Key>', handle_keypress)
    
    # Add window closing protocol
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing())
    
    # Create and pack widgets
    frame = ttk.Frame(root, padding="10")
    frame.pack(fill=tk.BOTH, expand=True)
    
    # Instructions label
    ttk.Label(frame, text="Select forex pairs (hold Ctrl/Cmd for multiple):").pack(pady=5)
    
    # Listbox for pairs
    pairs_listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, height=10)
    pairs_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
    
    # Add scrollbar
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=pairs_listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    pairs_listbox.config(yscrollcommand=scrollbar.set)
    
    # Add pairs to listbox
    for pair in available_pairs:
        pairs_listbox.insert(tk.END, pair)
    
    # Plot type selection
    plot_type = ttk.Combobox(frame, values=["Line Plot", "Candlestick"], state="readonly")
    plot_type.set("Line Plot")
    plot_type.pack(pady=5)
    
    # Plot button
    ttk.Button(frame, text="Plot Selected Pairs", command=plot_selected).pack(pady=10)
    
    # Start the GUI
    try:
        root.mainloop()
    except KeyboardInterrupt:
        on_closing()

if __name__ == "__main__":
    # Initialize database before starting GUI
    if not initialize_database():
        print("Failed to initialize database. Exiting...")
        sys.exit(1)
    
    try:
        create_gui()
    except KeyboardInterrupt:
        cleanup()
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}")
        cleanup()
        sys.exit(1)