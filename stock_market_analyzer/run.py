#!/usr/bin/env python3

import os
import sys
from pathlib import Path

def setup_environment():
    """Set up the Python path to include the project root."""
    # Get the absolute path of the project root directory
    project_root = Path(__file__).resolve().parent.parent
    
    # Add the project root to Python path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

def main():
    """Main entry point for the application."""
    setup_environment()
    
    # Import the main function after setting up the environment
    from stock_market_analyzer.main import main as app_main
    app_main()

if __name__ == "__main__":
    main() 