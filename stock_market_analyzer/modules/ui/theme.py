from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtCore import Qt

class DarkTheme:
    """Shared dark theme for the application."""
    
    @staticmethod
    def get_stylesheet():
        """Get the dark theme stylesheet."""
        return """
        QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QGroupBox {
            border: 1px solid #3c3c3c;
            border-radius: 5px;
            margin-top: 1ex;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px 0 3px;
            color: #ffffff;
        }
        QComboBox {
            background-color: #3c3c3c;
            border: 1px solid #3c3c3c;
            border-radius: 3px;
            padding: 5px;
            min-width: 6em;
        }
        QComboBox:hover {
            border: 1px solid #4c4c4c;
        }
        QComboBox::drop-down {
            border: none;
        }
        QPushButton {
            background-color: #3c3c3c;
            border: 1px solid #3c3c3c;
            border-radius: 3px;
            padding: 5px 10px;
            color: #ffffff;
        }
        QPushButton:hover {
            background-color: #4c4c4c;
            border: 1px solid #4c4c4c;
        }
        QPushButton:pressed {
            background-color: #2c2c2c;
        }
        QLineEdit {
            background-color: #3c3c3c;
            border: 1px solid #3c3c3c;
            border-radius: 3px;
            padding: 5px;
            color: #ffffff;
        }
        QLineEdit:focus {
            border: 1px solid #4c4c4c;
        }
        QTextEdit {
            background-color: #3c3c3c;
            border: 1px solid #3c3c3c;
            border-radius: 3px;
            color: #ffffff;
        }
        QTextEdit:focus {
            border: 1px solid #4c4c4c;
        }
        QTableWidget {
            background-color: #3c3c3c;
            border: 1px solid #3c3c3c;
            border-radius: 3px;
            color: #ffffff;
            gridline-color: #4c4c4c;
        }
        QTableWidget::item {
            padding: 5px;
        }
        QTableWidget::item:selected {
            background-color: #4c4c4c;
        }
        QHeaderView::section {
            background-color: #2b2b2b;
            color: #ffffff;
            padding: 5px;
            border: 1px solid #3c3c3c;
        }
        QDateEdit {
            background-color: #3c3c3c;
            border: 1px solid #3c3c3c;
            border-radius: 3px;
            padding: 5px;
            color: #ffffff;
        }
        QDateEdit:focus {
            border: 1px solid #4c4c4c;
        }
        QDateEdit::drop-down {
            border: none;
        }
        QCalendarWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QCalendarWidget QToolButton {
            background-color: #3c3c3c;
            color: #ffffff;
        }
        QCalendarWidget QMenu {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QCalendarWidget QAbstractItemView:enabled {
            background-color: #2b2b2b;
            color: #ffffff;
            selection-background-color: #4c4c4c;
            selection-color: #ffffff;
        }
        QLabel {
            color: #ffffff;
        }
        QLabel[status="error"] {
            color: #ff4444;
        }
        QLabel[status="success"] {
            color: #44ff44;
        }
        QScrollBar:vertical {
            border: none;
            background: #2b2b2b;
            width: 10px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #3c3c3c;
            min-height: 20px;
            border-radius: 3px;
        }
        QScrollBar::handle:vertical:hover {
            background: #4c4c4c;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        QScrollBar:horizontal {
            border: none;
            background: #2b2b2b;
            height: 10px;
            margin: 0px;
        }
        QScrollBar::handle:horizontal {
            background: #3c3c3c;
            min-width: 20px;
            border-radius: 3px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #4c4c4c;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        """
    
    @staticmethod
    def get_palette():
        """Get the dark theme palette."""
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Base, QColor(60, 60, 60))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(43, 43, 43))
        palette.setColor(QPalette.ColorRole.ToolTipText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Text, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Button, QColor(60, 60, 60))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 255, 255))
        palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
        return palette
    
    @staticmethod
    def apply_theme(widget):
        """Apply the dark theme to a widget."""
        widget.setStyleSheet(DarkTheme.get_stylesheet())
        widget.setPalette(DarkTheme.get_palette())
        
    @staticmethod
    def center_window(window):
        """Center a window on the screen."""
        screen = window.screen()
        if screen:
            screen_geometry = screen.geometry()
            window_geometry = window.geometry()
            x = (screen_geometry.width() - window_geometry.width()) // 2
            y = (screen_geometry.height() - window_geometry.height()) // 2
            window.move(x, y) 