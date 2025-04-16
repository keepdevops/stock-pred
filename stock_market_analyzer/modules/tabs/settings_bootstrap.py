import sys
import os
import logging
from PyQt6.QtWidgets import QApplication

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stock_market_analyzer.modules.tabs.settings_tab import SettingsTab

def main():
    """Main function for the settings tab process."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting settings tab process")
    
    # Create and show the settings tab
    app = QApplication(sys.argv)
    window = SettingsTab()
    window.setWindowTitle("Settings")
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 