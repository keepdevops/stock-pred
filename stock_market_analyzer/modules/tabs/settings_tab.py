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
        # Create main layout
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)
        
        # Create tab widget
        self.tab_widget = QTabWidget()
        
        # General Settings Tab
        general_tab = QWidget()
        general_layout = QVBoxLayout()
        
        # Theme settings
        theme_group = QGroupBox("Theme Settings")
        theme_layout = QVBoxLayout()
        
        # Color scheme
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color Scheme:"))
        self.color_scheme_combo = QComboBox()
        self.color_scheme_combo.addItems([
            "Light Theme",
            "Dark Theme",
            "High Contrast",
            "Protanopia",
            "Deuteranopia",
            "Tritanopia"
        ])
        self.color_scheme_combo.currentTextChanged.connect(self.on_settings_changed)
        color_layout.addWidget(self.color_scheme_combo)
        theme_layout.addLayout(color_layout)
        
        # Font settings
        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("Font Size:"))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setValue(12)
        self.font_size_spin.valueChanged.connect(self.on_settings_changed)
        font_layout.addWidget(self.font_size_spin)
        theme_layout.addLayout(font_layout)
        
        theme_group.setLayout(theme_layout)
        general_layout.addWidget(theme_group)
        
        # Data Settings
        data_group = QGroupBox("Data Settings")
        data_layout = QVBoxLayout()
        
        # Data path
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Data Directory:"))
        self.data_path_edit = QLineEdit()
        self.data_path_edit.textChanged.connect(self.on_settings_changed)
        path_layout.addWidget(self.data_path_edit)
        
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self.browse_data_path)
        path_layout.addWidget(browse_button)
        data_layout.addLayout(path_layout)
        
        # Cache settings
        cache_layout = QHBoxLayout()
        cache_layout.addWidget(QLabel("Cache Size (MB):"))
        self.cache_size_spin = QSpinBox()
        self.cache_size_spin.setRange(100, 10000)
        self.cache_size_spin.setValue(1000)
        self.cache_size_spin.valueChanged.connect(self.on_settings_changed)
        cache_layout.addWidget(self.cache_size_spin)
        data_layout.addLayout(cache_layout)
        
        data_group.setLayout(data_layout)
        general_layout.addWidget(data_group)
        
        general_tab.setLayout(general_layout)
        self.tab_widget.addTab(general_tab, "General")
        
        # Database Settings Tab
        db_tab = QWidget()
        db_layout = QVBoxLayout()
        
        # Database connection
        db_group = QGroupBox("Database Connection")
        db_group_layout = QVBoxLayout()
        
        # Database type
        db_type_layout = QHBoxLayout()
        db_type_layout.addWidget(QLabel("Database Type:"))
        self.db_type_combo = QComboBox()
        self.db_type_combo.addItems(["SQLite", "PostgreSQL", "MySQL", "DuckDB"])
        self.db_type_combo.currentTextChanged.connect(self.on_db_type_changed)
        db_type_layout.addWidget(self.db_type_combo)
        db_group_layout.addLayout(db_type_layout)
        
        # Connection settings form
        self.db_settings_widget = QWidget()
        self.db_settings_layout = QVBoxLayout()
        self.db_settings_widget.setLayout(self.db_settings_layout)
        db_group_layout.addWidget(self.db_settings_widget)
        
        db_group.setLayout(db_group_layout)
        db_layout.addWidget(db_group)
        
        db_tab.setLayout(db_layout)
        self.tab_widget.addTab(db_tab, "Database")
        
        # Trading Settings Tab
        trading_tab = QWidget()
        trading_layout = QVBoxLayout()
        
        # Trading configuration
        trading_group = QGroupBox("Trading Configuration")
        trading_group_layout = QVBoxLayout()
        
        # Trading mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Trading Mode:"))
        self.trading_mode_combo = QComboBox()
        self.trading_mode_combo.addItems(["Paper Trading", "Live Trading"])
        self.trading_mode_combo.currentTextChanged.connect(self.on_settings_changed)
        mode_layout.addWidget(self.trading_mode_combo)
        trading_group_layout.addLayout(mode_layout)
        
        # Risk management
        risk_layout = QHBoxLayout()
        risk_layout.addWidget(QLabel("Risk Level (%):"))
        self.risk_spin = QDoubleSpinBox()
        self.risk_spin.setRange(0.1, 10.0)
        self.risk_spin.setValue(2.0)
        self.risk_spin.valueChanged.connect(self.on_settings_changed)
        risk_layout.addWidget(self.risk_spin)
        trading_group_layout.addLayout(risk_layout)
        
        # Auto trading
        self.auto_trading_check = QCheckBox("Enable Auto Trading")
        self.auto_trading_check.stateChanged.connect(self.on_settings_changed)
        trading_group_layout.addWidget(self.auto_trading_check)
        
        trading_group.setLayout(trading_group_layout)
        trading_layout.addWidget(trading_group)
        
        trading_tab.setLayout(trading_layout)
        self.tab_widget.addTab(trading_tab, "Trading")
        
        # AI Settings Tab
        ai_tab = QWidget()
        ai_layout = QVBoxLayout()
        
        # AI configuration
        ai_group = QGroupBox("AI Configuration")
        ai_group_layout = QVBoxLayout()
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(["GPT-4", "GPT-3.5", "Custom Model"])
        self.model_combo.currentTextChanged.connect(self.on_settings_changed)
        model_layout.addWidget(self.model_combo)
        ai_group_layout.addLayout(model_layout)
        
        # Confidence threshold
        confidence_layout = QHBoxLayout()
        confidence_layout.addWidget(QLabel("Confidence Threshold:"))
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setValue(0.7)
        self.confidence_spin.setSingleStep(0.1)
        self.confidence_spin.valueChanged.connect(self.on_settings_changed)
        confidence_layout.addWidget(self.confidence_spin)
        ai_group_layout.addLayout(confidence_layout)
        
        ai_group.setLayout(ai_group_layout)
        ai_layout.addWidget(ai_group)
        
        ai_tab.setLayout(ai_layout)
        self.tab_widget.addTab(ai_tab, "AI")
        
        # Add tab widget to main layout
        self.main_layout.addWidget(self.tab_widget)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        button_layout.addWidget(save_button)
        
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self.reset_settings)
        button_layout.addWidget(reset_button)
        
        self.main_layout.addLayout(button_layout)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.main_layout.addWidget(self.status_label)
        
        # Initialize database settings
        self.on_db_type_changed(self.db_type_combo.currentText())
        
        # Load saved settings
        self.load_settings()
        
    def on_db_type_changed(self, db_type: str):
        """Update database settings form based on selected database type."""
        # Clear existing settings
        while self.db_settings_layout.count():
            item = self.db_settings_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        if db_type in ["PostgreSQL", "MySQL"]:
            # Server settings
            self.db_host_edit = QLineEdit()
            self.db_host_edit.setPlaceholderText("localhost")
            self.db_settings_layout.addWidget(QLabel("Host:"))
            self.db_settings_layout.addWidget(self.db_host_edit)
            
            self.db_port_spin = QSpinBox()
            self.db_port_spin.setRange(1, 65535)
            self.db_port_spin.setValue(5432 if db_type == "PostgreSQL" else 3306)
            self.db_settings_layout.addWidget(QLabel("Port:"))
            self.db_settings_layout.addWidget(self.db_port_spin)
            
            self.db_name_edit = QLineEdit()
            self.db_settings_layout.addWidget(QLabel("Database Name:"))
            self.db_settings_layout.addWidget(self.db_name_edit)
            
            self.db_user_edit = QLineEdit()
            self.db_settings_layout.addWidget(QLabel("Username:"))
            self.db_settings_layout.addWidget(self.db_user_edit)
            
            self.db_pass_edit = QLineEdit()
            self.db_pass_edit.setEchoMode(QLineEdit.EchoMode.Password)
            self.db_settings_layout.addWidget(QLabel("Password:"))
            self.db_settings_layout.addWidget(self.db_pass_edit)
        else:
            # File-based settings
            self.db_path_edit = QLineEdit()
            self.db_settings_layout.addWidget(QLabel("Database File:"))
            self.db_settings_layout.addWidget(self.db_path_edit)
            
            browse_button = QPushButton("Browse")
            browse_button.clicked.connect(self.browse_db_file)
            self.db_settings_layout.addWidget(browse_button)
            
    def test_settings(self):
        """Test the current settings."""
        try:
            # Test database connection
            db_type = self.db_type_combo.currentText()
            if db_type in ["PostgreSQL", "MySQL"]:
                # Test server connection
                host = self.db_host_edit.text()
                port = self.db_port_spin.value()
                db_name = self.db_name_edit.text()
                username = self.db_user_edit.text()
                password = self.db_pass_edit.text()
                
                # Create test connection
                if db_type == "PostgreSQL":
                    import psycopg2
                    conn = psycopg2.connect(
                        host=host,
                        port=port,
                        database=db_name,
                        user=username,
                        password=password
                    )
                else:
                    import mysql.connector
                    conn = mysql.connector.connect(
                        host=host,
                        port=port,
                        database=db_name,
                        user=username,
                        password=password
                    )
                conn.close()
                self.status_label.setText(f"{db_type} connection successful")
            else:
                # Test file-based database
                db_path = self.db_path_edit.text()
                if db_type == "SQLite":
                    import sqlite3
                    conn = sqlite3.connect(db_path)
                    conn.close()
                elif db_type == "DuckDB":
                    import duckdb
                    conn = duckdb.connect(db_path)
                    conn.close()
                self.status_label.setText(f"{db_type} connection successful")
                
            # Test data directory
            data_path = self.data_path_edit.text()
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            self.status_label.setText("All settings tested successfully")
            
        except Exception as e:
            self.status_label.setText(f"Test failed: {str(e)}")
            self.logger.error(f"Settings test failed: {str(e)}")
            QMessageBox.critical(self, "Test Failed", str(e))
            
    def reset_settings(self):
        """Reset settings to default values."""
        try:
            # Reset theme settings
            self.color_scheme_combo.setCurrentText("Light Theme")
            self.font_size_spin.setValue(12)
            
            # Reset data settings
            self.data_path_edit.setText(os.path.expanduser("~/stock_data"))
            self.cache_size_spin.setValue(1000)
            
            # Reset database settings
            self.db_type_combo.setCurrentText("SQLite")
            
            # Reset trading settings
            self.trading_mode_combo.setCurrentText("Paper Trading")
            self.risk_spin.setValue(2.0)
            self.auto_trading_check.setChecked(False)
            
            # Reset AI settings
            self.model_combo.setCurrentText("GPT-3.5")
            self.confidence_spin.setValue(0.7)
            
            self.status_label.setText("Settings reset to defaults")
            self.save_settings()  # Save the default values
            
        except Exception as e:
            self.status_label.setText(f"Reset failed: {str(e)}")
            self.logger.error(f"Settings reset failed: {str(e)}")
            
    def load_settings(self):
        """Load settings from configuration file."""
        try:
            config_path = os.path.join(os.path.expanduser("~"), ".stock_analyzer", "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    settings = json.load(f)
                    
                # Load theme settings
                self.color_scheme_combo.setCurrentText(settings.get('theme', {}).get('color_scheme', "Light Theme"))
                self.font_size_spin.setValue(settings.get('theme', {}).get('font_size', 12))
                
                # Load data settings
                self.data_path_edit.setText(settings.get('data', {}).get('path', os.path.expanduser("~/stock_data")))
                self.cache_size_spin.setValue(settings.get('data', {}).get('cache_size', 1000))
                
                # Load database settings
                db_settings = settings.get('database', {})
                self.db_type_combo.setCurrentText(db_settings.get('type', "SQLite"))
                if self.db_type_combo.currentText() in ["PostgreSQL", "MySQL"]:
                    self.db_host_edit.setText(db_settings.get('host', "localhost"))
                    self.db_port_spin.setValue(db_settings.get('port', 5432))
                    self.db_name_edit.setText(db_settings.get('database', ""))
                    self.db_user_edit.setText(db_settings.get('username', ""))
                    self.db_pass_edit.setText(db_settings.get('password', ""))
                else:
                    self.db_path_edit.setText(db_settings.get('path', ""))
                    
                # Load trading settings
                trading_settings = settings.get('trading', {})
                self.trading_mode_combo.setCurrentText(trading_settings.get('mode', "Paper Trading"))
                self.risk_spin.setValue(trading_settings.get('risk', 2.0))
                self.auto_trading_check.setChecked(trading_settings.get('auto_trading', False))
                
                # Load AI settings
                ai_settings = settings.get('ai', {})
                self.model_combo.setCurrentText(ai_settings.get('model', "GPT-3.5"))
                self.confidence_spin.setValue(ai_settings.get('confidence', 0.7))
                
                self.status_label.setText("Settings loaded successfully")
                
        except Exception as e:
            self.status_label.setText(f"Error loading settings: {str(e)}")
            self.logger.error(f"Error loading settings: {str(e)}")
            
    def save_settings(self):
        """Save settings to configuration file."""
        try:
            settings = {
                'theme': {
                    'color_scheme': self.color_scheme_combo.currentText(),
                    'font_size': self.font_size_spin.value()
                },
                'data': {
                    'path': self.data_path_edit.text(),
                    'cache_size': self.cache_size_spin.value()
                },
                'database': {
                    'type': self.db_type_combo.currentText()
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
            
            # Add database-specific settings
            if self.db_type_combo.currentText() in ["PostgreSQL", "MySQL"]:
                settings['database'].update({
                    'host': self.db_host_edit.text(),
                    'port': self.db_port_spin.value(),
                    'database': self.db_name_edit.text(),
                    'username': self.db_user_edit.text(),
                    'password': self.db_pass_edit.text()
                })
            else:
                settings['database']['path'] = self.db_path_edit.text()
                
            # Ensure config directory exists
            config_dir = os.path.join(os.path.expanduser("~"), ".stock_analyzer")
            os.makedirs(config_dir, exist_ok=True)
            
            # Save settings
            config_path = os.path.join(config_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(settings, f, indent=4)
                
            self.unsaved_changes = False
            self.status_label.setText("Settings saved successfully")
            
            # Notify other tabs about settings update
            self.message_bus.publish("Settings", "settings_updated", settings)
            
        except Exception as e:
            self.status_label.setText(f"Error saving settings: {str(e)}")
            self.logger.error(f"Error saving settings: {str(e)}")
            QMessageBox.critical(self, "Save Error", str(e))

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
            
    def browse_db_file(self):
        """Open file dialog to select database file."""
        try:
            path = QFileDialog.getOpenFileName(
                self,
                "Select Database File",
                os.path.expanduser("~"),
                "Database Files (*.db);;All Files (*)"
            )[0]
            
            if path:
                self.db_path_edit.setText(path)
                self.status_label.setText(f"Selected file: {path}")
                
        except Exception as e:
            self.logger.error(f"Error browsing database file: {e}")
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