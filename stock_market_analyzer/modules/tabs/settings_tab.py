import sys
import os
import logging
from typing import Any, Dict, Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QLineEdit, QGroupBox, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from stock_market_analyzer.modules.message_bus import MessageBus

class SettingsTab(QWidget):
    """Settings tab for the stock market analyzer."""
    
    def __init__(self, parent=None):
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
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # Database settings
        db_group = QGroupBox("Database Settings")
        db_layout = QVBoxLayout()
        
        # Database type
        db_type_layout = QHBoxLayout()
        db_type_layout.addWidget(QLabel("Database Type:"))
        self.db_type_combo = QComboBox()
        self.db_type_combo.addItems(["SQLite", "PostgreSQL", "MySQL", "DuckDB"])
        self.db_type_combo.currentTextChanged.connect(self.on_settings_changed)
        db_type_layout.addWidget(self.db_type_combo)
        db_layout.addLayout(db_type_layout)
        
        # Database path/host
        db_path_layout = QHBoxLayout()
        db_path_layout.addWidget(QLabel("Database Path/Host:"))
        self.db_path_edit = QLineEdit()
        self.db_path_edit.textChanged.connect(self.on_settings_changed)
        db_path_layout.addWidget(self.db_path_edit)
        db_layout.addLayout(db_path_layout)
        
        # Database port
        db_port_layout = QHBoxLayout()
        db_port_layout.addWidget(QLabel("Port:"))
        self.db_port_spin = QSpinBox()
        self.db_port_spin.setRange(1, 65535)
        self.db_port_spin.setValue(5432)
        self.db_port_spin.valueChanged.connect(self.on_settings_changed)
        db_port_layout.addWidget(self.db_port_spin)
        db_layout.addLayout(db_port_layout)
        
        # Database credentials
        db_cred_layout = QHBoxLayout()
        db_cred_layout.addWidget(QLabel("Username:"))
        self.db_user_edit = QLineEdit()
        self.db_user_edit.textChanged.connect(self.on_settings_changed)
        db_cred_layout.addWidget(self.db_user_edit)
        db_cred_layout.addWidget(QLabel("Password:"))
        self.db_pass_edit = QLineEdit()
        self.db_pass_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.db_pass_edit.textChanged.connect(self.on_settings_changed)
        db_cred_layout.addWidget(self.db_pass_edit)
        db_layout.addLayout(db_cred_layout)
        
        db_group.setLayout(db_layout)
        layout.addWidget(db_group)
        
        # Trading settings
        trading_group = QGroupBox("Trading Settings")
        trading_layout = QVBoxLayout()
        
        # Trading mode
        trading_mode_layout = QHBoxLayout()
        trading_mode_layout.addWidget(QLabel("Trading Mode:"))
        self.trading_mode_combo = QComboBox()
        self.trading_mode_combo.addItems(["Paper Trading", "Live Trading"])
        self.trading_mode_combo.currentTextChanged.connect(self.on_settings_changed)
        trading_mode_layout.addWidget(self.trading_mode_combo)
        trading_layout.addLayout(trading_mode_layout)
        
        # Risk settings
        risk_layout = QHBoxLayout()
        risk_layout.addWidget(QLabel("Max Risk per Trade (%):"))
        self.risk_spin = QDoubleSpinBox()
        self.risk_spin.setRange(0.1, 10.0)
        self.risk_spin.setValue(2.0)
        self.risk_spin.setDecimals(1)
        self.risk_spin.valueChanged.connect(self.on_settings_changed)
        risk_layout.addWidget(self.risk_spin)
        trading_layout.addLayout(risk_layout)
        
        # Auto-trading
        auto_trading_layout = QHBoxLayout()
        self.auto_trading_check = QCheckBox("Enable Auto-Trading")
        self.auto_trading_check.stateChanged.connect(self.on_settings_changed)
        auto_trading_layout.addWidget(self.auto_trading_check)
        trading_layout.addLayout(auto_trading_layout)
        
        trading_group.setLayout(trading_layout)
        layout.addWidget(trading_group)
        
        # AI settings
        ai_group = QGroupBox("AI Settings")
        ai_layout = QVBoxLayout()
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("AI Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["GPT-4", "GPT-3.5", "Claude"])
        self.model_combo.currentTextChanged.connect(self.on_settings_changed)
        model_layout.addWidget(self.model_combo)
        ai_layout.addLayout(model_layout)
        
        # Confidence threshold
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence Threshold:"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setValue(0.7)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.valueChanged.connect(self.on_settings_changed)
        confidence_layout.addWidget(self.confidence_spin)
        ai_layout.addLayout(confidence_layout)
        
        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)
        
        # Save button
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)
        
        # Status label
        self.status_label = QLabel("Settings loaded")
        layout.addWidget(self.status_label)
        
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