import tkinter as tk
import sys
from pathlib import Path
from app.gui import StockGUI

def check_environment():
    """Check if all required directories exist"""
    required_dirs = [
        Path("templates"),
        Path("data")
    ]
    
    for directory in required_dirs:
        if not directory.exists():
            directory.mkdir(parents=True)
            print(f"Created directory: {directory}")

def main():
    # Check environment
    check_environment()
    
    # Create main window
    root = tk.Tk()
    root.title("Stock Market Analyzer")
    root.geometry("1200x800")
    
    try:
        # Create GUI
        app = StockGUI(root)
        
        # Start the application
        root.mainloop()
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 