"""
Launch the Stock Market Analyzer from stock_new.
Run from this directory or from stock_new (python stock_market_analyzer/main.py).
"""
import sys
import os
from pathlib import Path

# Ensure stock_new is on the path and is the current working directory
_here = Path(__file__).resolve().parent
_stock_new = _here.parent
_stock_new_str = str(_stock_new)
if _stock_new_str not in sys.path:
    sys.path.insert(0, _stock_new_str)
os.chdir(_stock_new_str)

# Run the real app
from main import main

if __name__ == "__main__":
    main()
