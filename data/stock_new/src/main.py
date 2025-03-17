import sys
from pathlib import Path
import tkinter as tk

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Local imports
from src.modules.gui import StockGUI
from src.modules.database import DatabaseConnector
from src.modules.data_adapter import DataAdapter
from src.modules.stock_ai_agent import StockAIAgent

def find_databases(base_path: Path = Path.cwd() / "data") -> list[Path]:
    """Find all .duckdb files in the data directory."""
    return list(base_path.glob("**/*.duckdb"))

def initialize_components(root: tk.Tk) -> StockGUI:
    """Initialize all system components and return GUI instance."""
    db_connector = DatabaseConnector()
    data_adapter = DataAdapter()
    ai_agent = StockAIAgent(data_adapter)
    return StockGUI(root, db_connector, data_adapter, ai_agent)

def main():
    # Create root window
    root = tk.Tk()
    root.title("Stock Market Analyzer")
    root.geometry("1200x800")
    
    # Set theme
    try:
        root.tk.call("tk", "use", "clam")
    except tk.TclError:
        print("Warning: Clam theme not available")
    
    # Initialize GUI
    gui = initialize_components(root)
    
    # Start event loop
    root.mainloop()

if __name__ == "__main__":
    main() 