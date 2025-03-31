#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def setup_environment():
    """Set up the Python path to include the project root."""
    # Get the absolute path of the project root directory (parent of stock_market_analyzer)
    project_root = Path(__file__).resolve().parent.parent
    
    # Add both the project root and the package directory to Python path
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(project_root / "stock_market_analyzer"))

if __name__ == "__main__":
    setup_environment()
    
    # Import and run the main function after setting up the environment
    from stock_market_analyzer.main import main
    main() 