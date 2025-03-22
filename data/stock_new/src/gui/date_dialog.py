from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton, 
                           QLabel, QCalendarWidget, QComboBox)
from PyQt5.QtCore import QDate
from datetime import datetime, timedelta

class DateRangeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Date Range")
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Preset periods
        preset_layout = QHBoxLayout()
        preset_layout.addWidget(QLabel("Preset Periods:"))
        self.preset_combo = QComboBox()
        self.preset_combo.addItems([
            "Custom",
            "Last Week",
            "Last Month",
            "Last 3 Months",
            "Last 6 Months",
            "Year to Date",
            "Last Year",
            "Last 2 Years",
            "Last 5 Years"
        ])
        self.preset_combo.currentTextChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self.preset_combo)
        layout.addLayout(preset_layout)

        # Calendars
        dates_layout = QHBoxLayout()
        
        # Start date
        start_layout = QVBoxLayout()
        start_layout.addWidget(QLabel("Start Date:"))
        self.start_calendar = QCalendarWidget()
        start_layout.addWidget(self.start_calendar)
        dates_layout.addLayout(start_layout)
        
        # End date
        end_layout = QVBoxLayout()
        end_layout.addWidget(QLabel("End Date:"))
        self.end_calendar = QCalendarWidget()
        end_layout.addWidget(self.end_calendar)
        dates_layout.addLayout(end_layout)
        
        layout.addLayout(dates_layout)

        # Buttons
        buttons_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        buttons_layout.addWidget(ok_button)
        buttons_layout.addWidget(cancel_button)
        layout.addLayout(buttons_layout)

        # Set default dates
        self.end_calendar.setSelectedDate(QDate.currentDate())
        self.start_calendar.setSelectedDate(QDate.currentDate().addDays(-30))

    def on_preset_changed(self, text):
        today = QDate.currentDate()
        if text == "Last Week":
            self.start_calendar.setSelectedDate(today.addDays(-7))
        elif text == "Last Month":
            self.start_calendar.setSelectedDate(today.addMonths(-1))
        elif text == "Last 3 Months":
            self.start_calendar.setSelectedDate(today.addMonths(-3))
        elif text == "Last 6 Months":
            self.start_calendar.setSelectedDate(today.addMonths(-6))
        elif text == "Year to Date":
            self.start_calendar.setSelectedDate(QDate(today.year(), 1, 1))
        elif text == "Last Year":
            self.start_calendar.setSelectedDate(today.addYears(-1))
        elif text == "Last 2 Years":
            self.start_calendar.setSelectedDate(today.addYears(-2))
        elif text == "Last 5 Years":
            self.start_calendar.setSelectedDate(today.addYears(-5))
        self.end_calendar.setSelectedDate(today)

    def get_dates(self):
        """Get selected dates as Python datetime objects."""
        start_date = self.start_calendar.selectedDate().toPyDate()
        end_date = self.end_calendar.selectedDate().toPyDate()
        return datetime.combine(start_date, datetime.min.time()), \
               datetime.combine(end_date, datetime.max.time()) 