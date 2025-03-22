from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                           QComboBox, QLabel, QColorDialog, QFormLayout,
                           QScrollArea, QWidget, QMessageBox)
from PyQt5.QtCore import Qt
from .color_schemes import ColorScheme, ColorSchemeManager

class ColorSchemeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Color Scheme Settings")
        self.setModal(True)
        self.color_manager = ColorSchemeManager()
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Scheme selection
        scheme_layout = QHBoxLayout()
        scheme_layout.addWidget(QLabel("Select Color Scheme:"))
        self.scheme_combo = QComboBox()
        self.scheme_combo.addItems([scheme.value for scheme in ColorScheme])
        self.scheme_combo.currentTextChanged.connect(self.on_scheme_changed)
        scheme_layout.addWidget(self.scheme_combo)
        layout.addLayout(scheme_layout)
        
        # Preview area
        preview_scroll = QScrollArea()
        preview_widget = QWidget()
        self.preview_layout = QFormLayout(preview_widget)
        preview_scroll.setWidget(preview_widget)
        preview_scroll.setWidgetResizable(True)
        layout.addWidget(preview_scroll)
        
        # Custom scheme controls
        custom_layout = QHBoxLayout()
        self.save_custom_button = QPushButton("Save Custom Scheme")
        self.save_custom_button.clicked.connect(self.save_custom_scheme)
        custom_layout.addWidget(self.save_custom_button)
        layout.addLayout(custom_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(apply_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)
        
        # Initialize preview
        self.update_preview()

    def update_preview(self):
        """Update the color preview area."""
        # Clear existing preview
        while self.preview_layout.count():
            item = self.preview_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Get current scheme
        scheme_name = ColorScheme(self.scheme_combo.currentText())
        colors = self.color_manager.schemes[scheme_name]
        
        # Add color previews
        for name, color in colors.items():
            label = QLabel(name.replace('_', ' ').title())
            color_button = QPushButton()
            color_button.setStyleSheet(f"background-color: {color}; min-width: 100px;")
            color_button.setEnabled(scheme_name == ColorScheme.CUSTOM)
            if scheme_name == ColorScheme.CUSTOM:
                color_button.clicked.connect(lambda checked, n=name: self.pick_color(n))
            self.preview_layout.addRow(label, color_button)

    def on_scheme_changed(self, scheme_name):
        """Handle scheme selection change."""
        self.update_preview()
        self.save_custom_button.setEnabled(scheme_name == ColorScheme.CUSTOM.value)

    def pick_color(self, color_name):
        """Open color picker for custom scheme."""
        scheme = self.color_manager.schemes[ColorScheme.CUSTOM]
        current_color = QColor(scheme[color_name])
        color = QColorDialog.getColor(current_color, self)
        
        if color.isValid():
            scheme[color_name] = color.name()
            self.update_preview()

    def save_custom_scheme(self):
        """Save current custom scheme."""
        try:
            scheme = self.color_manager.schemes[ColorScheme.CUSTOM]
            self.color_manager.save_custom_scheme("custom", scheme)
            QMessageBox.information(self, "Success", "Custom color scheme saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save custom scheme: {str(e)}")

    def get_selected_scheme(self):
        """Get the selected color scheme."""
        return ColorScheme(self.scheme_combo.currentText()) 