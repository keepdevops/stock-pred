from PyQt5.QtCore import QObject, pyqtSignal, QTimer, QDateTime
import asyncio
import logging
from typing import Dict, Any, Optional
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
    """Manager for async operations."""
    
    # Signals for task status
    task_started = pyqtSignal(str)  # task name
    task_completed = pyqtSignal(str, Any)  # task name, result
    task_error = pyqtSignal(str, str)  # task name, error message
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.logger = logging.getLogger(__name__)
        self.tasks = {}
        
    async def run_task(self, task_name: str, coro):
        """Run an async task and emit status signals."""
        try:
            self.task_started.emit(task_name)
            result = await coro
            self.task_completed.emit(task_name, result)
            return result
            
        except Exception as e:
            self.logger.error(f"Error in task {task_name}: {e}")
            self.task_error.emit(task_name, str(e))
            raise
            
    def create_task(self, task_name: str, coro):
        """Create and run a new async task."""
        loop = asyncio.get_event_loop()
        task = loop.create_task(self.run_task(task_name, coro))
        self.tasks[task_name] = task
        return task
        
    def cancel_task(self, task_name: str):
        """Cancel a running task."""
        if task_name in self.tasks:
            self.tasks[task_name].cancel()
            del self.tasks[task_name]
            
    def cancel_all_tasks(self):
        """Cancel all running tasks."""
        for task_name in list(self.tasks.keys()):
            self.cancel_task(task_name) 