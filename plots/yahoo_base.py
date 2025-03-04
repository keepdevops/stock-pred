import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk

class BaseStockApp:
    # Add plot_stock as a class attribute
    plot_stock = None
    
    def __init__(self, master):
        # Window setup
        self.master = master
        self.master.title("Stock Analysis App")

        # Create variables
        self.ticker_var = tk.StringVar()
        self.period_var = tk.StringVar(value="1y")
        self.bb_std_var = tk.StringVar(value="2")
        self.title_var = tk.StringVar()

        # Create frames
        self.input_frame = tk.Frame(self.master)
        self.input_frame.grid(row=0, column=0, padx=10, pady=10, sticky='n')
        
        self.plot_frame = tk.Frame(self.master)
        self.plot_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
        
        # Configure grid weights
        self.master.grid_columnconfigure(1, weight=1)
        self.master.grid_rowconfigure(0, weight=1)

        # Assign the plot_stock method
        self.plot_stock = self._plot_stock

        # Setup UI
        self.create_menus()
        self.setup_controls()

    def create_menus(self):
        self.menubar = tk.Menu(self.master)
        self.master.config(menu=self.menubar)

        # Create File menu
        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Exit", command=self.master.quit)

        # Create Help menu
        self.help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=self.help_menu)
        self.help_menu.add_command(label="About", command=self.show_about)

    def setup_controls(self):
        # Create and layout widgets
        tk.Label(self.input_frame, text="Ticker Symbol:").grid(row=0, column=0, sticky='w')
        tk.Entry(self.input_frame, textvariable=self.ticker_var).grid(row=0, column=1)
        
        tk.Label(self.input_frame, text="Period:").grid(row=1, column=0, sticky='w')
        tk.Entry(self.input_frame, textvariable=self.period_var).grid(row=1, column=1)
        
        tk.Label(self.input_frame, text="BB Standard Dev:").grid(row=2, column=0, sticky='w')
        tk.Entry(self.input_frame, textvariable=self.bb_std_var).grid(row=2, column=1)
        
        tk.Label(self.input_frame, text="Chart Title:").grid(row=3, column=0, sticky='w')
        tk.Entry(self.input_frame, textvariable=self.title_var).grid(row=3, column=1)
        
        # Add Plot button
        tk.Button(
            self.input_frame, 
            text="Plot", 
            command=self.plot_stock
        ).grid(row=4, column=0, columnspan=2, pady=10)

    def _plot_stock(self):
        # Clear any existing plots in the frame
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
            
        # Get the stock data
        ticker = self.ticker_var.get().upper()
        if not ticker:
            return
            
        stock = yf.Ticker(ticker)
        data = stock.history(period=self.period_var.get())
        
        if data.empty:
            return
            
        # Create figure and axis
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        # Plot stock price
        ax.plot(data.index, data['Close'], label='Close Price')
        
        # Calculate and plot Bollinger Bands
        bb_std = float(self.bb_std_var.get())
        sma = data['Close'].rolling(window=20).mean()
        std = data['Close'].rolling(window=20).std()
        upper_band = sma + (std * bb_std)
        lower_band = sma - (std * bb_std)
        
        ax.plot(data.index, upper_band, 'r--', label='Upper BB')
        ax.plot(data.index, lower_band, 'r--', label='Lower BB')
        ax.plot(data.index, sma, 'g--', label='20-day SMA')
        
        # Set title and labels
        title = self.title_var.get() or f"{ticker} Stock Price"
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45)
        
        # Create canvas and add to frame
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def show_about(self):
        about_window = tk.Toplevel(self.master)
        about_window.title("About")
        about_window.geometry("300x100")
        
        label = tk.Label(about_window, text="Stock Analysis App\nVersion 1.0\n\nA simple stock analysis tool using Yahoo Finance data.")
        label.pack(padx=20, pady=20)

def main():
    root = tk.Tk()
    app = BaseStockApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 