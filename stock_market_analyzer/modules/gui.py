from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QProgressBar, QMessageBox, QFileDialog, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, QDateTime
from PyQt5.QtGui import QFont
import logging
import pandas as pd
from typing import Dict, Any, Optional

from .charts import StockChart, TechnicalIndicatorChart
from .realtime import RealTimeDataManager, AsyncTaskManager

class StockGUI(QMainWindow):
    def __init__(self, db, data_loader, ai_agent, trading_agent):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.db = db
        self.data_loader = data_loader
        self.ai_agent = ai_agent
        self.trading_agent = trading_agent
        
        # Initialize managers
        self.realtime_manager = RealTimeDataManager(self)
        self.async_manager = AsyncTaskManager(self)
        
        # Connect signals
        self.realtime_manager.price_update.connect(self.on_price_update)
        self.realtime_manager.indicator_update.connect(self.on_indicator_update)
        self.realtime_manager.error_occurred.connect(self.on_realtime_error)
        
        self.async_manager.task_started.connect(self.on_task_started)
        self.async_manager.task_completed.connect(self.on_task_completed)
        self.async_manager.task_error.connect(self.on_task_error)
        
        self.setup_gui()
        
    def setup_gui(self):
        """Set up the main GUI window and components."""
        self.setWindowTitle("Stock Market Analyzer")
        self.setMinimumSize(1200, 800)
        
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_data_tab(), "Data")
        self.tabs.addTab(self.create_analysis_tab(), "Analysis")
        self.tabs.addTab(self.create_charts_tab(), "Charts")
        self.tabs.addTab(self.create_trading_tab(), "Trading")
        self.tabs.addTab(self.create_models_tab(), "Models")
        self.tabs.addTab(self.create_settings_tab(), "Settings")
        
        # Add tabs to main layout
        main_layout.addWidget(self.tabs)
        
    def create_data_tab(self):
        """Set up the data management tab."""
        data_widget = QWidget()
        layout = QVBoxLayout(data_widget)
        
        # Data loading controls
        controls_layout = QHBoxLayout()
        
        # Symbol input
        symbol_label = QLabel("Symbol:")
        self.symbol_entry = QLineEdit()
        self.symbol_entry.setPlaceholderText("Enter stock symbol")
        controls_layout.addWidget(symbol_label)
        controls_layout.addWidget(self.symbol_entry)
        
        # Load button
        load_button = QPushButton("Load Data")
        load_button.clicked.connect(self.load_data)
        controls_layout.addWidget(load_button)
        
        # Real-time toggle
        self.realtime_button = QPushButton("Start Real-time")
        self.realtime_button.setCheckable(True)
        self.realtime_button.clicked.connect(self.toggle_realtime)
        controls_layout.addWidget(self.realtime_button)
        
        layout.addLayout(controls_layout)
        
        # Create splitter for charts and data
        splitter = QSplitter(Qt.Vertical)
        
        # Charts
        charts_widget = QWidget()
        charts_layout = QVBoxLayout(charts_widget)
        
        self.price_chart = StockChart()
        charts_layout.addWidget(self.price_chart)
        
        self.indicator_chart = TechnicalIndicatorChart()
        charts_layout.addWidget(self.indicator_chart)
        
        splitter.addWidget(charts_widget)
        
        # Data table
        self.data_table = QTableWidget()
        self.data_table.setColumnCount(6)
        self.data_table.setHorizontalHeaderLabels([
            "Date", "Open", "High", "Low", "Close", "Volume"
        ])
        self.data_table.horizontalHeader().setStretchLastSection(True)
        splitter.addWidget(self.data_table)
        
        layout.addWidget(splitter)
        
        self.tabs.addTab(data_widget, "Data")
        
        return data_widget
        
    def toggle_realtime(self, checked: bool):
        """Toggle real-time updates for the current symbol."""
        symbol = self.symbol_entry.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Warning", "Please enter a stock symbol")
            self.realtime_button.setChecked(False)
            return
            
        if checked:
            self.realtime_manager.start_updates(symbol)
            self.realtime_button.setText("Stop Real-time")
        else:
            self.realtime_manager.stop_updates(symbol)
            self.realtime_button.setText("Start Real-time")
            
    def on_price_update(self, symbol: str, price: float):
        """Handle real-time price updates."""
        try:
            # Update price chart
            timestamp = QDateTime.currentDateTime()
            self.price_chart.add_real_time_point(timestamp, price)
            
            # Update data table
            if self.data_table.rowCount() > 0:
                latest_row = self.data_table.rowCount() - 1
                self.data_table.setItem(latest_row, 4, QTableWidgetItem(f"{price:.2f}"))
                
        except Exception as e:
            self.logger.error(f"Error handling price update: {e}")
            
    def on_indicator_update(self, symbol: str, indicators: Dict[str, float]):
        """Handle real-time indicator updates."""
        try:
            timestamp = QDateTime.currentDateTime()
            self.indicator_chart.add_real_time_point(timestamp, indicators.get('close', 0), indicators)
            
        except Exception as e:
            self.logger.error(f"Error handling indicator update: {e}")
            
    def on_realtime_error(self, error: str):
        """Handle real-time update errors."""
        self.logger.error(f"Real-time error: {error}")
        QMessageBox.warning(self, "Real-time Error", error)
        
    def on_task_started(self, task_name: str):
        """Handle task start."""
        self.logger.info(f"Task started: {task_name}")
        
    def on_task_completed(self, task_name: str, result: Any):
        """Handle task completion."""
        self.logger.info(f"Task completed: {task_name}")
        
    def on_task_error(self, task_name: str, error: str):
        """Handle task errors."""
        self.logger.error(f"Task error: {task_name} - {error}")
        QMessageBox.critical(self, "Task Error", f"Error in {task_name}: {error}")
        
    def load_data(self):
        """Load stock data for the specified symbol."""
        try:
            symbol = self.symbol_entry.text().strip().upper()
            if not symbol:
                QMessageBox.warning(self, "Warning", "Please enter a stock symbol")
                return
                
            # Show loading message
            self.analysis_text.setText(f"Loading data for {symbol}...")
            
            # Create async task for data loading
            async def load_data_task():
                data = await self.data_loader.load_data_async(symbol)
                if data is None or data.empty:
                    raise ValueError(f"No data available for symbol {symbol}")
                return data
                
            self.async_manager.create_task(f"load_data_{symbol}", load_data_task())
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load data: {str(e)}")
            self.analysis_text.setText(f"Error: {str(e)}")
            
    def update_data_display(self, data: pd.DataFrame):
        """Update the data display with new data."""
        try:
            # Update data table
            self.data_table.setRowCount(len(data))
            
            for i, (_, row) in enumerate(data.iterrows()):
                self.data_table.setItem(i, 0, QTableWidgetItem(row['date'].strftime('%Y-%m-%d')))
                self.data_table.setItem(i, 1, QTableWidgetItem(f"{row['open']:.2f}"))
                self.data_table.setItem(i, 2, QTableWidgetItem(f"{row['high']:.2f}"))
                self.data_table.setItem(i, 3, QTableWidgetItem(f"{row['low']:.2f}"))
                self.data_table.setItem(i, 4, QTableWidgetItem(f"{row['close']:.2f}"))
                self.data_table.setItem(i, 5, QTableWidgetItem(f"{row['volume']:,.0f}"))
                
            # Update charts
            self.price_chart.update_data(data)
            self.indicator_chart.update_data(data)
            
        except Exception as e:
            self.logger.error(f"Error updating data display: {e}")
            QMessageBox.critical(self, "Error", f"Failed to update display: {str(e)}")
            
    def closeEvent(self, event):
        """Handle application close."""
        try:
            # Stop all real-time updates
            self.realtime_manager.stop_updates(self.symbol_entry.text().strip().upper())
            
            # Cancel all async tasks
            self.async_manager.cancel_all_tasks()
            
            super().closeEvent(event)
            
        except Exception as e:
            self.logger.error(f"Error during close: {e}")
            super().closeEvent(event)

    def create_analysis_tab(self):
        """Create the analysis tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Analysis controls
        controls_layout = QHBoxLayout()
        
        # Analysis type selection
        self.analysis_type = QComboBox()
        self.analysis_type.addItems(['Technical', 'Fundamental', 'Sentiment'])
        controls_layout.addWidget(QLabel("Analysis Type:"))
        controls_layout.addWidget(self.analysis_type)
        
        # Run analysis button
        self.run_analysis_btn = QPushButton("Run Analysis")
        self.run_analysis_btn.clicked.connect(self.run_analysis)
        controls_layout.addWidget(self.run_analysis_btn)
        
        layout.addLayout(controls_layout)
        
        # Analysis results
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        layout.addWidget(self.analysis_text)
        
        return tab
        
    def create_charts_tab(self):
        """Create the charts tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Chart controls
        controls_layout = QHBoxLayout()
        
        # Chart type selection
        self.chart_type = QComboBox()
        self.chart_type.addItems(['Price', 'Volume', 'Technical Indicators'])
        controls_layout.addWidget(QLabel("Chart Type:"))
        controls_layout.addWidget(self.chart_type)
        
        # Update chart button
        self.update_chart_btn = QPushButton("Update Chart")
        self.update_chart_btn.clicked.connect(self.update_chart)
        controls_layout.addWidget(self.update_chart_btn)
        
        layout.addLayout(controls_layout)
        
        # Create chart widgets
        self.price_chart = StockChart()
        self.technical_chart = TechnicalIndicatorChart()
        
        # Add charts to layout
        layout.addWidget(self.price_chart)
        layout.addWidget(self.technical_chart)
        
        return tab

    def create_trading_tab(self):
        """Set up the trading tab."""
        trading_widget = QWidget()
        layout = QVBoxLayout(trading_widget)
        
        # Trading controls
        controls_layout = QHBoxLayout()
        
        # Position size input
        position_label = QLabel("Position Size:")
        self.position_entry = QLineEdit()
        self.position_entry.setPlaceholderText("Enter position size")
        controls_layout.addWidget(position_label)
        controls_layout.addWidget(self.position_entry)
        
        # Buy/Sell buttons
        buy_button = QPushButton("Buy")
        buy_button.clicked.connect(self.execute_buy)
        sell_button = QPushButton("Sell")
        sell_button.clicked.connect(self.execute_sell)
        
        controls_layout.addWidget(buy_button)
        controls_layout.addWidget(sell_button)
        
        layout.addLayout(controls_layout)
        
        # Trading history table
        self.trading_table = QTableWidget()
        self.trading_table.setColumnCount(5)
        self.trading_table.setHorizontalHeaderLabels([
            "Date", "Type", "Symbol", "Size", "Price"
        ])
        self.trading_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.trading_table)
        
        self.tabs.addTab(trading_widget, "Trading")
        
        return trading_widget
        
    def create_models_tab(self):
        """Set up the model management tab."""
        model_widget = QWidget()
        layout = QVBoxLayout(model_widget)
        
        # Model selection
        selection_layout = QHBoxLayout()
        model_label = QLabel("Active Model:")
        self.model_combo = QComboBox()
        self.model_combo.addItems(self.ai_agent.get_available_models())
        self.model_combo.currentTextChanged.connect(self.on_model_change)
        selection_layout.addWidget(model_label)
        selection_layout.addWidget(self.model_combo)
        
        layout.addLayout(selection_layout)
        
        # Model parameters
        param_group = QGroupBox("Model Parameters")
        param_layout = QVBoxLayout()
        self.param_widgets = {}
        self.create_param_widgets(param_layout)
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Training controls
        train_group = QGroupBox("Training")
        train_layout = QVBoxLayout()
        
        # Training parameters
        param_layout = QHBoxLayout()
        
        # Epochs
        epochs_label = QLabel("Epochs:")
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(100)
        param_layout.addWidget(epochs_label)
        param_layout.addWidget(self.epochs_spin)
        
        # Batch size
        batch_label = QLabel("Batch Size:")
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(32)
        param_layout.addWidget(batch_label)
        param_layout.addWidget(self.batch_spin)
        
        # Sequence length (for LSTM)
        seq_label = QLabel("Sequence Length:")
        self.seq_spin = QSpinBox()
        self.seq_spin.setRange(1, 100)
        self.seq_spin.setValue(10)
        param_layout.addWidget(seq_label)
        param_layout.addWidget(self.seq_spin)
        
        train_layout.addLayout(param_layout)
        
        # Training buttons
        button_layout = QHBoxLayout()
        train_button = QPushButton("Train Model")
        train_button.clicked.connect(self.train_model)
        tune_button = QPushButton("Tune Hyperparameters")
        tune_button.clicked.connect(self.tune_model)
        predict_button = QPushButton("Make Prediction")
        predict_button.clicked.connect(self.make_prediction)
        
        button_layout.addWidget(train_button)
        button_layout.addWidget(tune_button)
        button_layout.addWidget(predict_button)
        train_layout.addLayout(button_layout)
        
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        # Model status
        status_group = QGroupBox("Model Status")
        status_layout = QVBoxLayout()
        self.model_status = QTextEdit()
        self.model_status.setReadOnly(True)
        status_layout.addWidget(self.model_status)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        self.tabs.addTab(model_widget, "Models")
        
        return model_widget
        
    def create_settings_tab(self):
        """Set up the settings tab."""
        settings_widget = QWidget()
        layout = QVBoxLayout(settings_widget)
        
        # API settings
        api_group = QGroupBox("API Settings")
        api_layout = QVBoxLayout()
        
        api_key_label = QLabel("API Key:")
        self.api_key_entry = QLineEdit()
        self.api_key_entry.setEchoMode(QLineEdit.Password)
        api_layout.addWidget(api_key_label)
        api_layout.addWidget(self.api_key_entry)
        
        api_group.setLayout(api_layout)
        layout.addWidget(api_group)
        
        # Save button
        save_button = QPushButton("Save Settings")
        save_button.clicked.connect(self.save_settings)
        layout.addWidget(save_button)
        
        self.tabs.addTab(settings_widget, "Settings")
        
        return settings_widget
        
    def create_param_widgets(self, layout):
        """Create parameter input widgets based on the active model."""
        # Clear existing widgets
        for widget in self.param_widgets.values():
            layout.removeWidget(widget)
        self.param_widgets.clear()
        
        # Get model parameters
        model = self.ai_agent.get_active_model()
        if model is None:
            return
            
        # Create widgets for each parameter
        for param, value in model.__dict__.items():
            if not param.startswith('_'):
                param_layout = QHBoxLayout()
                label = QLabel(f"{param}:")
                
                if isinstance(value, int):
                    widget = QSpinBox()
                    widget.setRange(1, 1000)
                    widget.setValue(value)
                elif isinstance(value, float):
                    widget = QDoubleSpinBox()
                    widget.setRange(0.0, 1000.0)
                    widget.setValue(value)
                    widget.setSingleStep(0.1)
                else:
                    widget = QLineEdit()
                    widget.setText(str(value))
                    
                param_layout.addWidget(label)
                param_layout.addWidget(widget)
                layout.addLayout(param_layout)
                self.param_widgets[param] = widget
                
    def on_model_change(self, model_id):
        """Handle model selection change."""
        try:
            self.ai_agent.set_active_model(model_id)
            self.create_param_widgets(self.param_widgets['layout'])
            self.update_model_status()
        except Exception as e:
            self.logger.error(f"Error changing model: {e}")
            QMessageBox.critical(self, "Error", f"Failed to change model: {str(e)}")
            
    def run_analysis(self):
        """Run analysis on the loaded data."""
        try:
            # Get the current data from the data table
            data = []
            for row in range(self.data_table.rowCount()):
                values = [self.data_table.item(row, col).text() for col in range(self.data_table.columnCount())]
                data.append({
                    'date': pd.to_datetime(values[0]),
                    'open': float(values[1]),
                    'high': float(values[2]),
                    'low': float(values[3]),
                    'close': float(values[4]),
                    'volume': float(values[5].replace(',', ''))
                })
            
            if not data:
                QMessageBox.warning(self, "Warning", "No data available for analysis")
                return
                
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Run analysis
            results = self.ai_agent.analyze(df)
            self.update_analysis_display(results)
            
        except Exception as e:
            self.logger.error(f"Error running analysis: {e}")
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
            
    def execute_buy(self):
        """Execute a buy order."""
        try:
            # Get the current symbol
            symbol = self.symbol_entry.text().strip().upper()
            if not symbol:
                QMessageBox.warning(self, "Warning", "Please enter a stock symbol")
                return
                
            # Get the current price from the data table
            if self.data_table.rowCount() == 0:
                QMessageBox.warning(self, "Warning", "No data available. Please load data first.")
                return
                
            # Get the most recent price
            latest_row = self.data_table.rowCount() - 1
            latest_values = [self.data_table.item(latest_row, col).text() for col in range(self.data_table.columnCount())]
            current_price = float(latest_values[4])  # Close price
            
            # Get current balance
            current_balance = self.trading_agent.get_balance()
            
            # Calculate maximum position size based on risk management
            max_position_size = self.trading_agent.calculate_position_size(symbol, current_price)
            
            # Get position size from user
            size_str = self.position_entry.text().strip()
            if not size_str:
                # If no size specified, use the calculated maximum
                size = max_position_size
                self.position_entry.clear()
                self.position_entry.insert(0, f"{size:.2f}")
            else:
                try:
                    size = float(size_str)
                    if size <= 0:
                        QMessageBox.warning(self, "Warning", "Position size must be greater than 0")
                        return
                    if size > max_position_size:
                        QMessageBox.warning(
                            self, 
                            "Warning", 
                            f"Position size too large. Maximum allowed: {max_position_size:.2f} shares\n"
                            f"Current balance: ${current_balance:.2f}\n"
                            f"Current price: ${current_price:.2f}"
                        )
                        return
                except ValueError:
                    messagebox.showwarning("Warning", "Please enter a valid number for position size")
                    return
            
            # Place the buy order
            if self.trading_agent.place_buy_order(symbol, size, current_price):
                self.update_trading_display()
                messagebox.showinfo(
                    "Success",
                    f"Buy order placed successfully:\n"
                    f"Symbol: {symbol}\n"
                    f"Size: {size:.2f} shares\n"
                    f"Price: ${current_price:.2f}\n"
                    f"Total: ${(size * current_price):.2f}\n"
                    f"Remaining balance: ${(current_balance - size * current_price):.2f}"
                )
            else:
                messagebox.showerror("Error", "Failed to place buy order")
            
        except Exception as e:
            self.logger.error(f"Error executing buy order: {e}")
            messagebox.showerror("Error", f"Buy order failed: {str(e)}")
            
    def execute_sell(self):
        """Execute a sell order."""
        try:
            # Get the current symbol
            symbol = self.symbol_entry.get().strip().upper()
            if not symbol:
                messagebox.showwarning("Warning", "Please enter a stock symbol")
                return
                
            # Get the current price from the data tree
            if not self.data_tree.get_children():
                messagebox.showwarning("Warning", "No data available. Please load data first.")
                return
                
            # Get the most recent price
            latest_item = self.data_tree.get_children()[-1]
            latest_values = self.data_tree.item(latest_item)['values']
            current_price = float(latest_values[4])  # Close price
            
            # Get position size
            size_str = self.position_entry.get().strip()
            if not size_str:
                messagebox.showwarning("Warning", "Please enter a position size")
                return
                
            try:
                size = float(size_str)
                if size <= 0:
                    messagebox.showwarning("Warning", "Position size must be greater than 0")
                    return
            except ValueError:
                messagebox.showwarning("Warning", "Please enter a valid number for position size")
                return
                
            self.trading_agent.place_sell_order(symbol, size, current_price)
            self.update_trading_display()
            
        except Exception as e:
            self.logger.error(f"Error executing sell order: {e}")
            messagebox.showerror("Error", f"Sell order failed: {str(e)}")
            
    def save_settings(self):
        """Save the current settings."""
        try:
            # Implementation depends on what settings need to be saved
            messagebox.showinfo("Success", "Settings saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving settings: {e}")
            messagebox.showerror("Error", f"Failed to save settings: {str(e)}")
            
    def update_analysis_display(self, results):
        """Update the analysis display with new results."""
        self.analysis_text.delete('1.0', tk.END)
        self.analysis_text.insert('1.0', str(results))
        
    def update_trading_display(self):
        """Update the trading history display."""
        # Clear existing items
        for item in self.trading_tree.get_children():
            self.trading_tree.delete(item)
            
        # Add trading history
        history = self.trading_agent.get_trading_history()
        for trade in history:
            self.trading_tree.insert('', 'end', values=trade)

    def update_model_status(self):
        """Update the model status display."""
        model = self.ai_agent.get_active_model()
        if model is None:
            self.model_status.setText("No model selected")
            return
            
        status = f"Model Type: {model.__class__.__name__}\n"
        status += f"Parameters:\n"
        for param, value in model.__dict__.items():
            if not param.startswith('_'):
                status += f"  {param}: {value}\n"
                
        self.model_status.setText(status)
        
    def show_progress_dialog(self, title, message):
        """Show a progress dialog."""
        dialog = QMessageBox(self)
        dialog.setWindowTitle(title)
        dialog.setText(message)
        dialog.setStandardButtons(QMessageBox.NoButton)
        dialog.show()
        return dialog

    def train_model(self):
        """Train the active model."""
        try:
            # Get model parameters
            model = self.ai_agent.get_active_model()
            if model is None:
                QMessageBox.warning(self, "Warning", "No model selected")
                return
            
            # Get training parameters
            epochs = self.epochs_spin.value()
            batch_size = self.batch_spin.value()
            
            # Train the model
            self.ai_agent.train_model(model, epochs, batch_size)
            
            # Update model status
            self.update_model_status()
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            QMessageBox.critical(self, "Error", f"Training failed: {str(e)}")
            
    def tune_model(self):
        """Tune the hyperparameters of the active model."""
        try:
            # Get model parameters
            model = self.ai_agent.get_active_model()
            if model is None:
                QMessageBox.warning(self, "Warning", "No model selected")
                return
            
            # Tune the model
            self.ai_agent.tune_model(model)
            
            # Update model status
            self.update_model_status()
            
        except Exception as e:
            self.logger.error(f"Error tuning model: {e}")
            QMessageBox.critical(self, "Error", f"Tuning failed: {str(e)}")

    def update_chart(self):
        """Update the charts based on the selected chart type."""
        try:
            if self.current_data is None:
                self.show_error("No data available to display")
                return
                
            chart_type = self.chart_type.currentText()
            
            if chart_type == 'Price':
                self.price_chart.update_data(self.current_data)
                self.price_chart.show()
                self.technical_chart.hide()
            elif chart_type == 'Volume':
                # Create a copy of the data with volume as the main metric
                volume_data = self.current_data.copy()
                volume_data['close'] = volume_data['volume']
                self.price_chart.update_data(volume_data)
                self.price_chart.show()
                self.technical_chart.hide()
            elif chart_type == 'Technical Indicators':
                # Calculate technical indicators
                data = self.current_data.copy()
                data['ma5'] = data['close'].rolling(window=5).mean()
                data['ma20'] = data['close'].rolling(window=20).mean()
                data['rsi'] = self.calculate_rsi(data['close'])
                
                self.technical_chart.update_data(data)
                self.price_chart.hide()
                self.technical_chart.show()
                
        except Exception as e:
            self.logger.error(f"Error updating chart: {e}")
            self.show_error(f"Error updating chart: {str(e)}")
            
    def calculate_rsi(self, prices, period=14):
        """Calculate the Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def make_prediction(self):
        """Make predictions using the active model."""
        try:
            if self.current_data is None:
                QMessageBox.warning(self, "Warning", "No data available for prediction")
                return
                
            # Get predictions
            predictions = self.ai_agent.predict(self.current_data)
            
            if predictions is not None:
                # Update the chart with predictions
                self.price_chart.add_predictions(predictions)
                
                # Show prediction results
                latest_pred = predictions[-1][0]
                latest_actual = self.current_data['close'].iloc[-1]
                prediction_text = f"Latest Prediction: ${latest_pred:.2f}\n"
                prediction_text += f"Latest Actual: ${latest_actual:.2f}\n"
                prediction_text += f"Difference: ${(latest_pred - latest_actual):.2f}"
                
                self.model_status.append(prediction_text)
                
            else:
                QMessageBox.warning(self, "Warning", "Failed to make predictions")
                
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            QMessageBox.critical(self, "Error", f"Prediction failed: {str(e)}") 