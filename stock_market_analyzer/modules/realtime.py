from PyQt5.QtCore import QObject, pyqtSignal, QTimer, QDateTime
import asyncio
import logging
from typing import Dict, Any, Optional, Coroutine
import pandas as pd
import yfinance as yf

class RealTimeDataManager(QObject):
    """Manager for real-time data updates and async operations."""
    
    # Signals for data updates
    price_update = pyqtSignal(str, float)  # symbol, price
    indicator_update = pyqtSignal(str, dict)  # symbol, indicators
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.tickers: Dict[str, yf.Ticker] = {}
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_data)
        self.update_interval = 5000  # 5 seconds
        
    def start_updates(self, symbol: str):
        """Start real-time updates for a symbol."""
        try:
            if symbol not in self.tickers:
                self.tickers[symbol] = yf.Ticker(symbol)
                self.logger.info(f"Started real-time updates for {symbol}")
                
            if not self.update_timer.isActive():
                self.update_timer.start(self.update_interval)
                
        except Exception as e:
            self.logger.error(f"Error starting updates for {symbol}: {e}")
            self.error_occurred.emit(f"Failed to start updates: {str(e)}")
            
    def stop_updates(self, symbol: str):
        """Stop real-time updates for a symbol."""
        try:
            if symbol in self.tickers:
                del self.tickers[symbol]
                self.logger.info(f"Stopped real-time updates for {symbol}")
                
            if not self.tickers and self.update_timer.isActive():
                self.update_timer.stop()
                
        except Exception as e:
            self.logger.error(f"Error stopping updates for {symbol}: {e}")
            self.error_occurred.emit(f"Failed to stop updates: {str(e)}")
            
    def update_data(self):
        """Update data for all active symbols."""
        for symbol, ticker in self.tickers.items():
            try:
                # Get real-time data
                info = ticker.info
                current_price = info.get('regularMarketPrice')
                
                if current_price:
                    # Calculate indicators
                    indicators = self.calculate_indicators(ticker)
                    
                    # Emit signals
                    self.price_update.emit(symbol, current_price)
                    self.indicator_update.emit(symbol, indicators)
                    
            except Exception as e:
                self.logger.error(f"Error updating {symbol}: {e}")
                self.error_occurred.emit(f"Update failed for {symbol}: {str(e)}")
                
    def calculate_indicators(self, ticker: yf.Ticker) -> Dict[str, float]:
        """Calculate technical indicators for a ticker."""
        try:
            # Get historical data
            hist = ticker.history(period='1d', interval='1m')
            
            if hist.empty:
                return {}
                
            # Calculate indicators
            indicators = {}
            
            # Moving averages
            indicators['ma5'] = hist['Close'].rolling(window=5).mean().iloc[-1]
            indicators['ma20'] = hist['Close'].rolling(window=20).mean().iloc[-1]
            
            # RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs)).iloc[-1]
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return {}
            
    def set_update_interval(self, interval: int):
        """Set the update interval in milliseconds."""
        self.update_interval = interval
        if self.update_timer.isActive():
            self.update_timer.setInterval(interval)

class AsyncTaskManager(QObject):
    """Manages asynchronous tasks in the application."""
    
    task_started = pyqtSignal(str)
    task_completed = pyqtSignal(str, object)
    task_error = pyqtSignal(str, str)
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.tasks: Dict[str, asyncio.Task] = {}
        self.loop = None
        self._setup_event_loop()
        
    def _setup_event_loop(self):
        """Set up the asyncio event loop."""
        try:
            if self.loop is None or self.loop.is_closed():
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.logger.info("Event loop setup completed")
        except Exception as e:
            self.logger.error(f"Error setting up event loop: {e}")
            raise
            
    def create_task(self, task_name: str, coro: Coroutine) -> None:
        """Create and start a new async task."""
        try:
            if task_name in self.tasks:
                self.logger.warning(f"Task {task_name} already exists, cancelling previous task")
                self.tasks[task_name].cancel()
                del self.tasks[task_name]
                
            # Ensure event loop is running
            if not self.loop.is_running():
                self._setup_event_loop()
                
            # Create and store the task
            task = self.loop.create_task(coro)
            self.tasks[task_name] = task
            
            # Add callback for task completion
            task.add_done_callback(lambda t: self._handle_task_completion(task_name, t))
            
            self.logger.info(f"Creating new task: {task_name}")
            self.task_started.emit(task_name)
            
        except Exception as e:
            self.logger.error(f"Error creating task {task_name}: {e}")
            self.task_error.emit(task_name, str(e))
            
    def _handle_task_completion(self, task_name: str, task: asyncio.Task) -> None:
        """Handle task completion."""
        try:
            if task_name in self.tasks:
                del self.tasks[task_name]
                
            if task.cancelled():
                self.logger.info(f"Task {task_name} was cancelled")
                return
                
            if task.exception():
                error = task.exception()
                self.logger.error(f"Task {task_name} failed: {error}")
                self.task_error.emit(task_name, str(error))
            else:
                result = task.result()
                self.logger.info(f"Task {task_name} completed")
                self.logger.info(f"Task {task_name} completed successfully with result type: {type(result)}")
                self.task_completed.emit(task_name, result)
                
        except Exception as e:
            self.logger.error(f"Error handling task completion for {task_name}: {e}")
            self.task_error.emit(task_name, str(e))
            
    def cleanup(self):
        """Clean up resources."""
        try:
            # Cancel all tasks
            for task_name, task in self.tasks.items():
                self.logger.info(f"Cancelling task: {task_name}")
                task.cancel()
            self.tasks.clear()
            
            # Stop the event loop
            if self.loop and not self.loop.is_closed():
                self.loop.stop()
                self.loop.close()
                self.loop = None
                
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}") 