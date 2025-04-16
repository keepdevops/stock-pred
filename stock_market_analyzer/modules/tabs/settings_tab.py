import sys
import os
import json
import logging
from typing import Any, Dict, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QGroupBox, QMessageBox,
    QFileDialog, QTabWidget, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, QTimer
from modules.tabs.base_tab import BaseTab
from ..message_bus import MessageBus

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class SettingsTab(BaseTab):
    """Settings tab for the stock market analyzer."""
    
    def __init__(self, parent=None):
        """Initialize the Settings tab."""
        super().__init__(parent)
        self.message_bus = MessageBus()
        self.logger = logging.getLogger(__name__)
        self.unsaved_changes = False
        self.setup_ui()
        
        # Set up auto-save timer
        self.auto_save_timer = QTimer(self)
        self.auto_save_timer.timeout.connect(self.auto_save_settings)
        self.auto_save_timer.start(30000)  # Auto-save every 30 seconds
        
    def setup_ui(self):
        """Setup the settings tab UI."""
        # Create main layout if it doesn't exist
        if not hasattr(self, 'main_layout'):
            self.main_layout = QVBoxLayout()
            self.setLayout(self.main_layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # Create scroll area for each tab
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        # Settings tab
        settings_tab = QWidget()
        settings_layout = QVBoxLayout()
        
        # Add settings UI elements
        self.data_path_label = QLabel("Data Path:")
        settings_layout.addWidget(self.data_path_label)
        
        self.data_path_edit = QLineEdit()
        settings_layout.addWidget(self.data_path_edit)
        
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_data_path)
        settings_layout.addWidget(browse_button)
        
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        settings_layout.addWidget(save_button)
        
        settings_tab.setLayout(settings_layout)
        
        # Add tabs to tab widget
        self.tab_widget.addTab(settings_tab, "Settings")
        
        # Add tab widget to main layout
        self.main_layout.addWidget(self.tab_widget)
        
        # Subscribe to message bus
        self.message_bus.subscribe("Settings", self.handle_message)
        
        self.logger.info("Settings tab initialized")
        
    def on_settings_changed(self):
        """Handle settings changes."""
        self.unsaved_changes = True
        self.status_label.setText("Settings changed - not saved")
        
    def validate_settings(self, settings: Dict) -> Optional[str]:
        """Validate settings and return error message if invalid."""
        try:
            # Validate database settings
            if settings['database']['type'] == 'SQLite':
                if not settings['database']['path']:
                    return "SQLite database path is required"
            else:
                if not settings['database']['path']:
                    return "Database host is required"
                if not settings['database']['username']:
                    return "Database username is required"
                    
            # Validate trading settings
            if settings['trading']['mode'] == 'Live Trading' and settings['trading']['auto_trading']:
                return "Auto-trading is not allowed in Live Trading mode"
                
            # Validate AI settings
            if settings['ai']['confidence'] < 0.1 or settings['ai']['confidence'] > 1.0:
                return "Confidence threshold must be between 0.1 and 1.0"
                
            return None
            
        except Exception as e:
            return f"Error validating settings: {str(e)}"
            
    def handle_message(self, sender: str, message_type: str, data: Any):
        """Handle incoming messages."""
        try:
            if message_type == "settings_updated":
                self.update_settings(data)
            elif message_type == "error":
                error_msg = f"Received error from {sender}: {data}"
                self.logger.error(error_msg)
                self.status_label.setText(f"Error: {data}")
                QMessageBox.critical(self, "Error", error_msg)
                
        except Exception as e:
            error_log = f"Error handling message in Settings tab: {str(e)}"
            self.logger.error(error_log)
            
    def update_settings(self, settings: dict):
        """Update settings from received data."""
        try:
            # Update database settings
            if 'database' in settings:
                db = settings['database']
                self.db_type_combo.setCurrentText(db.get('type', 'SQLite'))
                self.db_path_edit.setText(db.get('path', ''))
                self.db_port_spin.setValue(db.get('port', 5432))
                self.db_user_edit.setText(db.get('username', ''))
                self.db_pass_edit.setText(db.get('password', ''))
                
            # Update trading settings
            if 'trading' in settings:
                trading = settings['trading']
                self.trading_mode_combo.setCurrentText(trading.get('mode', 'Paper Trading'))
                self.risk_spin.setValue(trading.get('risk', 2.0))
                self.auto_trading_check.setChecked(trading.get('auto_trading', False))
                
            # Update AI settings
            if 'ai' in settings:
                ai = settings['ai']
                self.model_combo.setCurrentText(ai.get('model', 'GPT-4'))
                self.confidence_spin.setValue(ai.get('confidence', 0.7))
                
            self.unsaved_changes = False
            self.status_label.setText("Settings updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating settings: {str(e)}")
            self.message_bus.publish("Settings", "error", str(e))
            
    def save_settings(self):
        """Save current settings."""
        try:
            settings = {
                'database': {
                    'type': self.db_type_combo.currentText(),
                    'path': self.db_path_edit.text(),
                    'port': self.db_port_spin.value(),
                    'username': self.db_user_edit.text(),
                    'password': self.db_pass_edit.text()
                },
                'trading': {
                    'mode': self.trading_mode_combo.currentText(),
                    'risk': self.risk_spin.value(),
                    'auto_trading': self.auto_trading_check.isChecked()
                },
                'ai': {
                    'model': self.model_combo.currentText(),
                    'confidence': self.confidence_spin.value()
                }
            }
            
            # Validate settings
            error = self.validate_settings(settings)
            if error:
                QMessageBox.warning(self, "Invalid Settings", error)
                return
                
            # Publish settings update
            self.message_bus.publish("Settings", "settings_updated", settings)
            self.unsaved_changes = False
            self.status_label.setText("Settings saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {str(e)}")
            self.message_bus.publish("Settings", "error", str(e))
            
    def auto_save_settings(self):
        """Auto-save settings if there are unsaved changes."""
        if self.unsaved_changes:
            self.save_settings()
            
    def closeEvent(self, event):
        """Handle the close event."""
        if self.unsaved_changes:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Would you like to save them?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel
            )
            
            if reply == QMessageBox.StandardButton.Save:
                self.save_settings()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
                
        super().closeEvent(event)
            
    def publish_message(self, message_type: str, data: Any):
        """Publish a message to the message bus."""
        try:
            self.message_bus.publish("Settings", message_type, data)
        except Exception as e:
            error_log = f"Error publishing message from Settings tab: {str(e)}"
            self.logger.error(error_log)

    def browse_data_path(self):
        """Open file dialog to select data path."""
        try:
            path = QFileDialog.getExistingDirectory(
                self,
                "Select Data Path",
                os.path.expanduser("~"),
                QFileDialog.Option.ShowDirsOnly
            )
            
            if path:
                self.data_path_edit.setText(path)
                self.status_label.setText(f"Selected path: {path}")
                
        except Exception as e:
            self.logger.error(f"Error browsing path: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def load_settings(self):
        """Load settings from configuration file."""
        try:
            # TODO: Implement settings loading from config file
            # For now, set some default values
            self.data_path_edit.setText(os.path.expanduser("~/stock_data"))
            self.db_path_edit.setText("localhost")
            self.db_port_spin.setValue(5432)
            self.db_user_edit.setText("postgres")
            
            self.status_label.setText("Settings loaded")
            
        except Exception as e:
            self.logger.error(f"Error loading settings: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
    def save_settings(self):
        """Save settings to configuration file."""
        try:
            # TODO: Implement settings saving to config file
            self.status_label.setText("Settings saved")
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            self.status_label.setText(f"Error: {str(e)}")
            
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