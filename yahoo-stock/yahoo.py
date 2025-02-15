import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tkinter as tk
import tkinter.messagebox
from tkinter import ttk
import webbrowser

def plot_stock_history(ticker_symbol, root=None):
    try:
        # Fetch data for given stock for the last year
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period="1y")
        
        if hist.empty:
            if root:
                tk.messagebox.showerror("Error", f"No data found for ticker symbol: {ticker_symbol}")
            return

        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, 
                           subplot_titles=(f'{ticker_symbol} Closing Price', 'Volume'),
                           row_heights=[0.7, 0.3],
                           vertical_spacing=0.12)

        # Add price trace
        fig.add_trace(
            go.Scatter(x=hist.index, y=hist['Close'],
                      name='Close Price',
                      line=dict(color='blue')),
            row=1, col=1
        )

        # Add volume trace
        fig.add_trace(
            go.Bar(x=hist.index, y=hist['Volume'],
                  name='Volume',
                  marker=dict(color='gray')),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title=f'{ticker_symbol} Stock Analysis',
            showlegend=True,
            height=800,
            width=1200,
            template='plotly_white'
        )

        # Update y-axes labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)

        # Show plot in browser
        fig.show()

    except Exception as e:
        if root:
            tk.messagebox.showerror("Error", f"Failed to plot stock data: {str(e)}")

def main():
    root = tk.Tk()
    root.title("Stock Analysis")
    
    # Create input frame
    input_frame = tk.Frame(root)
    input_frame.pack(side=tk.TOP, pady=10)
    
    # Add label for clarity
    label = ttk.Label(input_frame, text="Stock Symbol:")
    label.pack(side=tk.LEFT, padx=5)
    
    # Add input field and button
    entry = ttk.Entry(input_frame, width=20)
    entry.pack(side=tk.LEFT, padx=5)
    entry.insert(0, "Enter ticker symbol...")  # Add placeholder text
    
    def plot_stock():
        ticker = entry.get().upper().strip()  # Remove whitespace
        if not ticker or ticker == "ENTER TICKER SYMBOL...":
            tk.messagebox.showwarning("Warning", "Please enter a valid ticker symbol")
            return
        try:
            # Add a loading message
            status_label.config(text=f"Loading data for {ticker}...")
            root.update()
            plot_stock_history(ticker, root)
            status_label.config(text="")  # Clear status after loading
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to plot stock data: {str(e)}")
            status_label.config(text="")  # Clear status on error
    
    plot_button = ttk.Button(input_frame, text="Plot Stock", command=plot_stock)
    plot_button.pack(side=tk.LEFT, padx=5)
    
    # Add example label
    example_label = ttk.Label(input_frame, text="(e.g., AAPL, MSFT, GOOGL)")
    example_label.pack(side=tk.LEFT, padx=5)
    
    # Add status label below
    status_label = ttk.Label(root, text="")
    status_label.pack(pady=5)
    
    # Add placeholder text behavior
    def on_entry_click(event):
        if entry.get() == "Enter ticker symbol...":
            entry.delete(0, tk.END)
            entry.config(foreground='black')
            
    def on_focus_out(event):
        if entry.get() == "":
            entry.insert(0, "Enter ticker symbol...")
            entry.config(foreground='grey')
    
    # Add Enter key binding
    def on_enter(event):
        plot_stock()
            
    entry.bind('<FocusIn>', on_entry_click)
    entry.bind('<FocusOut>', on_focus_out)
    entry.bind('<Return>', on_enter)  # Allow Enter key to trigger plot
    entry.config(foreground='grey')  # Set initial text color
    
    root.mainloop()

if __name__ == "__main__":
    main()
