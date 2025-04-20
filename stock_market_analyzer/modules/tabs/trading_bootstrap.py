import sys
import os
import signal
import logging
import traceback
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import from modules directly
from modules.message_bus import MessageBus
from modules.tabs.trading_tab import TradingTab

def handle_shutdown(signum, frame):
    """Handle shutdown signal."""
    print("Received shutdown signal")
    QApplication.quit()

def main():
    """Main function for the trading tab process."""
    try:
        # Set up signal handlers
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
        
        # Create application
        app = QApplication(sys.argv)
        
        # Create message bus
        message_bus = MessageBus()
        
        # Create and show the trading tab
        trading_tab = TradingTab(message_bus=message_bus)
        trading_tab.show()
        
        # Start the application
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Error in trading tab process: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main() 