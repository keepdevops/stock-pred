import tkinter as tk
from pathlib import Path
from modules.gui import StockGUI
from modules.database import DatabaseConnector
from modules.data_adapter import DataAdapter
from modules.stock_ai_agent import StockAIAgent

def find_databases(base_path: Path = Path.cwd()) -> list[Path]:
    """Find all .duckdb files in the base directory."""
    return list(base_path.glob("**/*.duckdb"))

def initialize_components(root: tk.Tk) -> StockGUI:
    """Initialize all system components and return GUI instance."""
    # Initialize core components
    db_connector = DatabaseConnector()
    data_adapter = DataAdapter()
    ai_agent = StockAIAgent(data_adapter)
    
    # Create and return GUI with dependencies
    return StockGUI(root, db_connector, data_adapter, ai_agent)

def main():
    # Create root window
    root = tk.Tk()
    root.title("Stock Market Analyzer")
    root.geometry("1200x800")
    
    # Set theme
    root.tk.call("tk", "use", "clam")
    
    # Initialize GUI
    gui = initialize_components(root)
    
    # Start event loop
    root.mainloop()

if __name__ == "__main__":
    main() 