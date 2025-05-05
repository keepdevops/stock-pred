from PyQt6.QtWidgets import (QFileDialog, QPushButton, QHBoxLayout, QVBoxLayout, 
                            QLabel, QMessageBox, QMenu, QTextEdit, QDockWidget,
                            QMainWindow, QApplication, QWidget)
from pathlib import Path
import pandas as pd
import polars as pl
from PyQt6.QtCore import QTimer, Qt, QThread, pyqtSignal
from datetime import datetime
import logging

class QTextEditLogger(logging.Handler):
    """Custom log handler that redirects logs to a QTextEdit widget"""
    def __init__(self, text_edit):
        super().__init__()
        self.text_edit = text_edit
        self.formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', 
                                         datefmt='%Y-%m-%d %H:%M:%S')
        
    def emit(self, record):
        """Format log record and append to QTextEdit"""
        try:
            msg = self.formatter.format(record)
            self.text_edit.append(msg)
        except Exception:
            self.handleError(record)

class FileHandler:
    def __init__(self):
        self.supported_formats = {
            '.csv': self.load_csv,
            '.json': self.load_json,
            '.parquet': self.load_parquet,
            '.db': self.load_sqlite,
            '.duckdb': self.load_duckdb
        }
    
    def load_file(self, file_path):
        suffix = Path(file_path).suffix.lower()
        if suffix in self.supported_formats:
            return self.supported_formats[suffix](file_path)
        raise ValueError(f"Unsupported file format: {suffix}")

    def load_csv(self, file_path):
        try:
            return pl.read_csv(file_path).to_pandas()
        except:
            return pd.read_csv(file_path)

    def load_json(self, file_path):
        return pd.read_json(file_path)

    def load_parquet(self, file_path):
        try:
            return pl.read_parquet(file_path).to_pandas()
        except:
            return pd.read_parquet(file_path)

    def load_sqlite(self, file_path):
        import sqlite3
        conn = sqlite3.connect(file_path)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        if len(tables) > 0:
            return pd.read_sql_query(f"SELECT * FROM {tables.iloc[0]['name']}", conn)
        return None

    def load_duckdb(self, file_path):
        import duckdb
        conn = duckdb.connect(file_path)
        tables = conn.execute("SHOW TABLES").fetchall()
        if tables:
            return conn.execute(f"SELECT * FROM {tables[0][0]}").df()
        return None 

class StockGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... existing initialization code ...
        
        self.file_handler = FileHandler()
        self.training_data = None
        self.setup_file_controls()
        self.setup_log_display()

    def setup_log_display(self):
        """Set up log display window"""
        # Create a dock widget for the logs
        self.log_dock = QDockWidget("Log Messages", self)
        self.log_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        
        # Create a text edit for logs
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        
        # Set the text edit as the dock widget's content
        self.log_dock.setWidget(self.log_text)
        
        # Add dock to main window
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.log_dock)
        
        # Create and install the log handler
        self.log_handler = QTextEditLogger(self.log_text)
        self.log_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        
        # Get the root logger and add our handler
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        
        # Add a clear button
        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.clear_log)
        
        # Add button to a layout at the bottom of the dock
        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_text)
        log_layout.addWidget(self.clear_log_button)
        
        # Create a widget to hold this layout
        log_widget = QWidget()
        log_widget.setLayout(log_layout)
        
        # Set as dock widget content
        self.log_dock.setWidget(log_widget)
        
        # Log that we've set up the display
        logging.info("Log display initialized in GUI")
    
    def clear_log(self):
        """Clear the log display"""
        self.log_text.clear()
        logging.info("Log cleared")

    def setup_file_controls(self):
        """Add file controls to the GUI"""
        # Create file control widget
        file_widget = QWidget()
        file_layout = QHBoxLayout(file_widget)
        
        # Add Open File button
        self.open_file_btn = QPushButton('Open Training Data', self)
        self.open_file_btn.clicked.connect(self.open_training_file)
        file_layout.addWidget(self.open_file_btn)
        
        # Add label to show selected file
        self.file_label = QLabel('No file selected', self)
        file_layout.addWidget(self.file_label)
        
        # Add Train Model button
        self.train_btn = QPushButton('Train Model', self)
        self.train_btn.clicked.connect(self.train_model)
        self.train_btn.setEnabled(False)  # Disabled until file is loaded
        file_layout.addWidget(self.train_btn)
        
        # Add to main layout (adjust based on your existing layout)
        # Find your main layout widget
        main_widget = self.centralWidget()
        main_layout = main_widget.layout()
        
        # Insert the file controls at the top
        main_layout.insertWidget(0, file_widget)

    def open_training_file(self):
        """Open file dialog and load training data"""
        file_filter = "Data Files (*.csv *.json *.parquet *.db *.duckdb);;All Files (*)"
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Training Data",
            "",
            file_filter
        )
        
        if file_path:
            try:
                # Load the data
                self.training_data = self.file_handler.load_file(file_path)
                
                if self.training_data is not None:
                    # Update UI
                    file_name = Path(file_path).name
                    self.file_label.setText(f"Loaded: {file_name}")
                    self.train_btn.setEnabled(True)
                    
                    # Show data summary
                    self.show_data_summary()
                    
                    QMessageBox.information(
                        self,
                        "Success",
                        f"Successfully loaded {file_name}\n"
                        f"Rows: {len(self.training_data)}\n"
                        f"Columns: {', '.join(self.training_data.columns)}"
                    )
                else:
                    QMessageBox.warning(
                        self,
                        "Warning",
                        "No data found in the selected file."
                    )
            
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Error loading file: {str(e)}"
                )

    def show_data_summary(self):
        """Show summary of loaded data"""
        if self.training_data is not None:
            # Create summary window
            summary = QMessageBox(self)
            summary.setWindowTitle("Data Summary")
            
            # Prepare summary text
            text = "Data Summary:\n\n"
            text += f"Total Rows: {len(self.training_data)}\n"
            text += f"Total Columns: {len(self.training_data.columns)}\n\n"
            text += "Columns:\n"
            
            for col in self.training_data.columns:
                text += f"- {col}: {self.training_data[col].dtype}\n"
            
            summary.setText(text)
            summary.exec_()

    def train_model(self):
        """Handle model training"""
        if self.training_data is None:
            QMessageBox.warning(self, "Warning", "Please load training data first.")
            return
        
        try:
            # Verify data structure
            required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
            missing_columns = required_columns - set(self.training_data.columns)
            
            if missing_columns:
                QMessageBox.warning(
                    self,
                    "Invalid Data",
                    f"Missing required columns: {', '.join(missing_columns)}"
                )
                return
            
            # Start training process
            QMessageBox.information(
                self,
                "Training Started",
                "Model training has begun. This may take a while..."
            )
            
            # Here you would call your model training code
            # self.model_trainer.train(self.training_data)
            
            # For now, just show a placeholder message
            QMessageBox.information(
                self,
                "Training Complete",
                "Model training completed successfully!"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error during training: {str(e)}"
            )

    def validate_training_data(self, df):
        """Validate the structure of training data"""
        required_columns = {'Open', 'High', 'Low', 'Close', 'Volume'}
        if not all(col in df.columns for col in required_columns):
            return False, "Missing required columns"
        
        # Check for numeric data
        for col in required_columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                return False, f"Column {col} must be numeric"
        
        return True, "Data is valid"

    def download_data(self):
        """Download historical data for selected tickers"""
        if not self.selected_tickers:
            QMessageBox.warning(self, "No Tickers Selected", "Please select at least one ticker.")
            return

        # Disable the fetch button
        self.fetch_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Downloading data...")

        # Get date range
        start_date = self.start_date_edit.date().toString('yyyy-MM-dd')
        end_date = self.end_date_edit.date().toString('yyyy-MM-dd')

        # Get selected processing mode
        mode = self.processing_mode_combo.currentText().lower()
        batch_size = int(self.batch_size_spin.value()) if mode == 'batch' else None

        # Create and start worker
        self.download_worker = DataDownloadWorker(
            self.ticker_manager,
            self.selected_tickers,
            start_date,
            end_date,
            mode=mode,
            batch_size=batch_size
        )
        self.download_worker.progress.connect(self.update_progress)
        self.download_worker.finished.connect(self.download_completed)
        self.download_worker.error.connect(self.handle_download_error)
        self.download_worker.save_completed.connect(self.handle_save_completed)
        self.download_worker.start()

    def update_progress(self, value, message):
        """Update progress bar and status message"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        QApplication.processEvents()  # Force UI update

    def download_completed(self, data):
        """Handle completion of data download"""
        if data:
            self.stock_data = data
            self.update_chart()
        else:
            self.status_label.setText("No data was downloaded.")
        
        # Re-enable the fetch button after a short delay
        QTimer.singleShot(1000, lambda: self.fetch_button.setEnabled(True))

    def handle_download_error(self, error_message):
        """Handle download errors"""
        self.status_label.setText(f"Error: {error_message}")
        QMessageBox.critical(self, "Download Error", error_message)
        # Re-enable the fetch button after a short delay
        QTimer.singleShot(1000, lambda: self.fetch_button.setEnabled(True))

    def handle_save_completed(self, save_dir):
        """Handle completion of data save"""
        self.status_label.setText(f"Data saved to: {save_dir}")
        QMessageBox.information(self, "Save Completed", f"Data saved successfully to {save_dir}")

    def export_data(self, format_type: str):
        """Export data in the specified format."""
        if not hasattr(self, 'current_data') or self.current_data.empty:
            QMessageBox.warning(self, "Export Error", "No data available to export")
            return

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stock_data_{timestamp}"
            
            if format_type == "csv":
                path = self.ticker_manager.export_to_csv(self.current_data, filename)
                msg = f"Data exported to CSV: {path}"
            
            elif format_type == "json":
                path = self.ticker_manager.export_to_json(self.current_data, filename)
                msg = f"Data exported to JSON: {path}"
            
            elif format_type == "sqlite":
                path = self.ticker_manager.export_to_sqlite(
                    self.current_data, filename, "stock_quotes"
                )
                msg = f"Data exported to SQLite: {path}"
            
            elif format_type == "duckdb":
                path = self.ticker_manager.export_to_duckdb(
                    self.current_data, filename, "stock_quotes"
                )
                msg = f"Data exported to DuckDB: {path}"
            
            elif format_type == "excel":
                path = self.ticker_manager.export_to_excel(self.current_data, filename)
                msg = f"Data exported to Excel: {path}"
            
            elif format_type == "parquet":
                path = self.ticker_manager.export_to_parquet(self.current_data, filename)
                msg = f"Data exported to Parquet: {path}"
            
            elif format_type == "arrow":
                path = self.ticker_manager.export_to_arrow(self.current_data, filename)
                msg = f"Data exported to Arrow: {path}"
            
            elif format_type == "hdf5":
                path = self.ticker_manager.export_to_hdf5(self.current_data, filename)
                msg = f"Data exported to HDF5: {path}"
            
            elif format_type == "polars":
                polars_df = self.ticker_manager.convert_to_polars(self.current_data)
                msg = f"Data converted to Polars DataFrame with {len(polars_df)} rows"
            
            QMessageBox.information(self, "Export Success", msg)
            self.status_label.setText(f"Export successful: {format_type}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
            self.status_label.setText(f"Export failed: {str(e)}")
            logging.error(f"Export error: {e}")

class DataDownloadWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    save_completed = pyqtSignal(str)  # New signal for save completion

    def __init__(self, ticker_manager, tickers, start_date, end_date, mode='sequential', batch_size=None):
        super().__init__()
        self.ticker_manager = ticker_manager
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.mode = mode
        self.batch_size = batch_size

    def run(self):
        try:
            total_tickers = len(self.tickers)
            processed_tickers = 0
            data = {}

            if self.mode == 'sequential':
                for ticker in self.tickers:
                    self.progress.emit(
                        int((processed_tickers / total_tickers) * 100),
                        f"Processing {ticker} ({processed_tickers + 1}/{total_tickers})"
                    )
                    try:
                        ticker_data = self.ticker_manager.get_historical_data(
                            [ticker],
                            self.start_date,
                            self.end_date,
                            mode='sequential',
                            clean_data=True
                        )
                        if ticker_data:
                            data.update(ticker_data)
                    except Exception as e:
                        self.error.emit(f"Error processing {ticker}: {str(e)}")
                    processed_tickers += 1

            elif self.mode == 'parallel':
                self.progress.emit(50, f"Processing all {total_tickers} tickers in parallel")
                data = self.ticker_manager.get_historical_data(
                    self.tickers,
                    self.start_date,
                    self.end_date,
                    mode='parallel',
                    clean_data=True
                )
                processed_tickers = len(data)

            elif self.mode == 'async':
                self.progress.emit(50, f"Processing all {total_tickers} tickers asynchronously")
                data = self.ticker_manager.get_historical_data(
                    self.tickers,
                    self.start_date,
                    self.end_date,
                    mode='async',
                    clean_data=True
                )
                processed_tickers = len(data)

            elif self.mode == 'batch':
                for i in range(0, len(self.tickers), self.batch_size):
                    batch = self.tickers[i:i + self.batch_size]
                    self.progress.emit(
                        int((i / total_tickers) * 100),
                        f"Processing batch {i//self.batch_size + 1} ({len(batch)} tickers)"
                    )
                    try:
                        batch_data = self.ticker_manager.get_historical_data(
                            batch,
                            self.start_date,
                            self.end_date,
                            mode='parallel',
                            clean_data=True
                        )
                        if batch_data:
                            data.update(batch_data)
                    except Exception as e:
                        self.error.emit(f"Error processing batch: {str(e)}")
                    processed_tickers += len(batch)

            if data:
                # Save the data
                self.progress.emit(90, "Saving data...")
                save_dir = self.ticker_manager.save_data(data)
                if save_dir:
                    self.save_completed.emit(save_dir)
                    self.progress.emit(100, f"Completed processing and saving {processed_tickers} tickers")
                else:
                    self.error.emit("Failed to save data")
            else:
                self.error.emit("No data was downloaded successfully")

            self.finished.emit(data)

        except Exception as e:
            self.error.emit(f"Download failed: {str(e)}") 