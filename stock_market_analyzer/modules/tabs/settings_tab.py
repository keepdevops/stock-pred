import sys
import os
import json
import logging
from typing import Any, Dict, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QGroupBox, QMessageBox,
    QFileDialog, QTabWidget, QScrollArea, QFrame,
    QRadioButton, QFormLayout
)
from PyQt6.QtCore import Qt, QTimer
from modules.tabs.base_tab import BaseTab
from ..message_bus import MessageBus
import uuid
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class SettingsTab(BaseTab):
    """Settings tab for configuring application settings."""
    
    def __init__(self, parent=None):
        """Initialize the Settings tab."""
        # Initialize attributes before parent __init__
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.settings_cache = {}
        self._ui_setup_done = False
        self.main_layout = None
        self.color_scheme_combo = None
        self.font_size_spin = None
        self.font_family_combo = None
        self.save_button = None
        self.reset_button = None
        self.status_label = None
        
        super().__init__(parent)
        
        # Setup UI after parent initialization
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the UI components."""
        try:
            # Clear the base layout
            while self.main_layout.count():
                item = self.main_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
                    
            self.main_layout.setSpacing(10)
            self.main_layout.setContentsMargins(10, 10, 10, 10)
            
            # Create settings group
            settings_group = QGroupBox("Application Settings")
            settings_layout = QVBoxLayout()
            
            # Color scheme
            color_layout = QHBoxLayout()
            color_layout.addWidget(QLabel("Color Scheme:"))
            self.color_scheme_combo = QComboBox()
            self.color_scheme_combo.addItems([
                "Default",
                "Dark",
                "Light",
                "High Contrast"
            ])
            color_layout.addWidget(self.color_scheme_combo)
            settings_layout.addLayout(color_layout)
            
            # Font settings
            font_layout = QHBoxLayout()
            font_layout.addWidget(QLabel("Font Size:"))
            self.font_size_spin = QSpinBox()
            self.font_size_spin.setRange(8, 24)
            self.font_size_spin.setValue(12)
            font_layout.addWidget(self.font_size_spin)
            
            font_layout.addWidget(QLabel("Font Family:"))
            self.font_family_combo = QComboBox()
            self.font_family_combo.addItems([
                "Arial",
                "Helvetica",
                "Times New Roman",
                "Courier New"
            ])
            font_layout.addWidget(self.font_family_combo)
            settings_layout.addLayout(font_layout)
            
            # Action buttons
            button_layout = QHBoxLayout()
            
            self.save_button = QPushButton("Save Settings")
            self.save_button.clicked.connect(self.save_settings)
            button_layout.addWidget(self.save_button)
            
            self.reset_button = QPushButton("Reset to Default")
            self.reset_button.clicked.connect(self.reset_settings)
            button_layout.addWidget(self.reset_button)
            
            settings_layout.addLayout(button_layout)
            settings_group.setLayout(settings_layout)
            self.main_layout.addWidget(settings_group)
            
            # Create status bar
            status_layout = QHBoxLayout()
            
            self.status_label = QLabel("Ready")
            self.status_label.setStyleSheet("color: green")
            status_layout.addWidget(self.status_label)
            
            self.main_layout.addLayout(status_layout)
            
            self._ui_setup_done = True
            
        except Exception as e:
            error_msg = f"Error setting up UI: {str(e)}"
            self.logger.error(error_msg)
            if self.status_label:
                self.status_label.setText(error_msg)
                
    def _setup_message_bus_impl(self):
        """Setup message bus subscriptions."""
        super()._setup_message_bus_impl()
        self.message_bus.subscribe("Settings", self.handle_message)
        
    def save_settings(self):
        """Save the current settings."""
        try:
            settings = {
                'color_scheme': self.color_scheme_combo.currentText(),
                'font_size': self.font_size_spin.value(),
                'font_family': self.font_family_combo.currentText()
            }
            
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            # Create request
            request = {
                'request_id': request_id,
                'settings': settings,
                'timestamp': datetime.now()
            }
            
            # Publish request
            self.message_bus.publish(
                "Settings",
                "save_settings_request",
                request
            )
            
            # Update UI
            self.status_label.setText("Settings saved successfully")
            
        except Exception as e:
            error_msg = f"Error saving settings: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def reset_settings(self):
        """Reset settings to default values."""
        try:
            # Reset UI elements
            self.color_scheme_combo.setCurrentText("Default")
            self.font_size_spin.setValue(12)
            self.font_family_combo.setCurrentText("Arial")
            
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            # Create request
            request = {
                'request_id': request_id,
                'timestamp': datetime.now()
            }
            
            # Publish request
            self.message_bus.publish(
                "Settings",
                "reset_settings_request",
                request
            )
            
            # Update UI
            self.status_label.setText("Settings reset to default")
            
        except Exception as e:
            error_msg = f"Error resetting settings: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "settings_response":
                self.handle_settings_response(sender, data)
            elif message_type == "error":
                self.status_label.setText(f"Error: {data.get('error', 'Unknown error')}")
                
        except Exception as e:
            error_msg = f"Error handling message: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def handle_settings_response(self, sender: str, data: Any):
        """Handle settings response."""
        try:
            request_id = data.get('request_id')
            settings = data.get('settings', {})
            
            if settings:
                # Update UI with new settings
                if 'color_scheme' in settings:
                    self.color_scheme_combo.setCurrentText(settings['color_scheme'])
                if 'font_size' in settings:
                    self.font_size_spin.setValue(settings['font_size'])
                if 'font_family' in settings:
                    self.font_family_combo.setCurrentText(settings['font_family'])
                    
                self.status_label.setText("Settings updated successfully")
                
        except Exception as e:
            error_msg = f"Error handling settings response: {str(e)}"
            self.logger.error(error_msg)
            self.status_label.setText(error_msg)
            
    def cleanup(self):
        """Cleanup resources."""
        try:
            super().cleanup()
            self.settings_cache.clear()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            
    def closeEvent(self, event):
        """Handle the close event."""
        self.cleanup()
        super().closeEvent(event)

def main():
    """Main function for the settings tab process."""
    # Ensure QApplication instance exists
    app = QApplication.instance() 
    if not app: 
        app = QApplication(sys.argv)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.info("Starting settings tab process")
    
    # Create and show the settings tab
    try:
        window = SettingsTab()
        window.setWindowTitle("Settings Tab")
        window.show()
    except Exception as e:
         logger.error(f"Failed to create or show SettingsTab window: {e}")
         logger.error(traceback.format_exc())
         sys.exit(1)

    if __name__ == "__main__":
        sys.exit(app.exec())

if __name__ == "__main__":
    import traceback
    main() 