import logging
from src.data.ticker_manager import TickerManager

logging.basicConfig(level=logging.INFO)

def main():
    manager = TickerManager()
    print("Test successful!")

if __name__ == "__main__":
    main() 