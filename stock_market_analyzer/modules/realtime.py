from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QDateTime
import asyncio
import logging
import time
from typing import Dict, Any, Optional, Coroutine, Callable
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

class RealTimeDataManager(QObject):
    """Manages real-time data updates and async tasks."""
    
    price_updated = pyqtSignal(str, float)
    indicators_updated = pyqtSignal(str, dict)
    error_occurred = pyqtSignal(str)
    task_started = pyqtSignal(str)
    task_completed = pyqtSignal(str, object)
    task_error = pyqtSignal(str, str)

    def __init__(self, data_loader, parent=None):
        """Initialize the RealTimeDataManager.
        
        Args:
            data_loader: The data loader instance to use for fetching data
            parent: Optional parent QObject
        """
        super().__init__(parent)
        self.data_loader = data_loader
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.active_symbols = set()
        self.tasks = {}
        self.logger = logging.getLogger(__name__)
        self.tickers: Dict[str, yf.Ticker] = {}
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_data)
        self.update_interval = 5000  # 5 seconds
        
        # Initialize asyncio event loop
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.logger.info("Event loop setup completed")
        except Exception as e:
            self.logger.error(f"Error setting up event loop: {e}")
            raise
            
    def start(self, symbol: str) -> bool:
        """Start real-time updates for a symbol."""
        try:
            self.logger.info(f"Starting real-time updates for {symbol}")
            
            # Check if symbol is already being tracked
            if symbol in self.tickers:
                self.logger.warning(f"Symbol {symbol} is already being tracked")
                return True
                
            # Create yfinance ticker
            ticker = yf.Ticker(symbol)
            
            # Verify ticker is valid
            if not ticker.info:
                self.logger.error(f"Invalid symbol: {symbol}")
                self.error_occurred.emit(f"Invalid symbol: {symbol}")
                return False
                
            # Add to tracking
            self.tickers[symbol] = ticker
            
            # Start update timer if not already running
            if not self.update_timer.isActive():
                self.update_timer.start(self.update_interval)
                self.logger.info("Started update timer")
                
            self.logger.info(f"Successfully started real-time updates for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting real-time updates for {symbol}: {e}")
            self.error_occurred.emit(f"Failed to start updates: {str(e)}")
            return False

    def stop(self, symbol: str) -> bool:
        """Stop real-time updates for a symbol."""
        try:
            self.logger.info(f"Stopping real-time updates for {symbol}")
            
            # Check if symbol is being tracked
            if symbol not in self.tickers:
                self.logger.warning(f"Symbol {symbol} is not being tracked")
                return True
                
            # Remove from tracking
            del self.tickers[symbol]
            
            # Stop update timer if no symbols left
            if not self.tickers and self.update_timer.isActive():
                self.update_timer.stop()
                self.logger.info("Stopped update timer")
                
            self.logger.info(f"Successfully stopped real-time updates for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping real-time updates for {symbol}: {e}")
            self.error_occurred.emit(f"Failed to stop updates: {str(e)}")
            return False

    def is_running(self, symbol: str) -> bool:
        """Check if real-time updates are running for a symbol."""
        return symbol in self.tickers

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
                    self.price_updated.emit(symbol, current_price)
                    self.indicators_updated.emit(symbol, indicators)
                    
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

    def create_task(self, task_id: str, task_func: Callable, *args, **kwargs) -> str:
        """Create a new task for real-time data operations."""
        try:
            self.logger.info(f"Creating new task: {task_id}")
            
            # Create task function with error handling
            async def task_wrapper():
                try:
                    # Check if task_func is a coroutine
                    if asyncio.iscoroutinefunction(task_func):
                        result = await task_func(*args, **kwargs)
                    else:
                        result = await asyncio.to_thread(task_func, *args, **kwargs)
                    self.task_completed.emit(task_id, result)
                except Exception as e:
                    self.logger.error(f"Task {task_id} failed: {str(e)}")
                    self.task_error.emit(task_id, str(e))
            
            # Add task to event loop
            task = self.loop.create_task(task_wrapper())
            self.tasks[task_id] = task
            self.task_started.emit(task_id)
            
            return task_id
            
        except Exception as e:
            self.logger.error(f"Error creating task {task_id}: {str(e)}")
            raise

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get the status of a task."""
        try:
            task = self.tasks.get(task_id)
            if task:
                return {
                    'id': task_id,
                    'status': 'running' if not task.done() else 'completed',
                    'result': task.result() if task.done() else None,
                    'exception': task.exception() if task.done() and task.exception() else None
                }
            return {'id': task_id, 'status': 'not_found'}
        except Exception as e:
            self.logger.error(f"Error getting task status: {str(e)}")
            return {'id': task_id, 'status': 'error', 'error': str(e)}

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        try:
            task = self.tasks.get(task_id)
            if task and not task.done():
                task.cancel()
                self.logger.info(f"Cancelled task: {task_id}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error cancelling task: {str(e)}")
            return False

    def _run_event_loop(self):
        """Run the event loop for a short time."""
        try:
            if self.loop and not self.loop.is_closed():
                self.loop.stop()
                self.loop.run_forever()
        except Exception as e:
            self.logger.error(f"Error running event loop: {e}")
            
    def cleanup(self):
        """Clean up resources."""
        try:
            self.logger.info("Starting RealTimeDataManager cleanup")
            
            # Stop the timer first to prevent new events
            if hasattr(self, 'timer'):
                if self.timer.isActive():
                    self.logger.info("Stopping timer")
                    self.timer.stop()
            
            # Cancel all tasks
            task_names = list(self.tasks.keys())
            if task_names:
                self.logger.info(f"Cancelling {len(task_names)} active tasks")
                for task_name in task_names:
                    try:
                        self.cancel_task(task_name)
                    except Exception as e:
                        self.logger.error(f"Error cancelling task {task_name}: {e}")
            
            # Stop the event loop
            if self.loop and not self.loop.is_closed():
                try:
                    self.logger.info("Stopping event loop")
                    self.loop.stop()
                    self.loop.close()
                    self.loop = None
                except Exception as e:
                    self.logger.error(f"Error stopping event loop: {e}")
            
            self.logger.info("RealTimeDataManager cleanup completed")
                
        except Exception as e:
            self.logger.error(f"Error during RealTimeDataManager cleanup: {e}")
            # Try to ensure critical cleanup happens even with errors
            try:
                if hasattr(self, 'timer') and self.timer.isActive():
                    self.timer.stop()
                
                if self.loop and not self.loop.is_closed():
                    self.loop.close()
                    
                self.tasks.clear()
            except:
                pass

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
        
        # Create a timer to run the event loop
        self.timer = QTimer()
        self.timer.timeout.connect(self._run_event_loop)
        self.timer.start(100)  # Run every 100ms
        
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
            
    def _run_event_loop(self):
        """Run the event loop for a short time."""
        try:
            if self.loop and not self.loop.is_closed():
                self.loop.stop()
                self.loop.run_forever()
        except Exception as e:
            self.logger.error(f"Error running event loop: {e}")
    
    def run_task(self, task_name: str, coro_or_func) -> None:
        """Run an async task with the given name.
        
        Args:
            task_name: Unique name to identify the task
            coro_or_func: Coroutine object or async function to run
        """
        try:
            # Handle both coroutine objects and async functions
            if asyncio.iscoroutine(coro_or_func):
                # It's already a coroutine object, use directly
                coro = coro_or_func
            elif asyncio.iscoroutinefunction(coro_or_func):
                # It's an async function, call it to get a coroutine
                coro = coro_or_func()
            elif callable(coro_or_func):
                # It's a regular function, assume it returns a coroutine when called
                coro = coro_or_func()
                if not asyncio.iscoroutine(coro):
                    raise TypeError(f"Function {coro_or_func.__name__} did not return a coroutine")
            else:
                raise TypeError("Expected a coroutine object or async function")
            
            return self.create_task(task_name, coro)
        except Exception as e:
            self.logger.error(f"Error preparing task {task_name}: {e}")
            self.task_error.emit(task_name, str(e))
            
    def create_task(self, task_name: str, coro: Coroutine) -> None:
        """Create and start a new async task."""
        try:
            # Ensure we have a coroutine object
            if not asyncio.iscoroutine(coro):
                raise TypeError("Expected a coroutine object, not a function reference")
            
            # Cancel existing task if it exists
            if task_name in self.tasks:
                self.logger.warning(f"Task {task_name} already exists, cancelling previous task")
                self._cancel_task(task_name)
            
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
            
    def _cancel_task(self, task_name: str) -> None:
        """Cancel a specific task."""
        try:
            if task_name in self.tasks:
                task = self.tasks[task_name]
                if not task.done():
                    task.cancel()
                del self.tasks[task_name]
                self.logger.info(f"Cancelled task: {task_name}")
        except Exception as e:
            self.logger.error(f"Error cancelling task {task_name}: {e}")
            
    def _handle_task_completion(self, task_name: str, task: asyncio.Task) -> None:
        """Handle task completion."""
        try:
            # Remove task from tracking
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
            self.logger.info("Starting AsyncTaskManager cleanup")
            
            # Stop the timer first to prevent new events
            if hasattr(self, 'timer'):
                if self.timer.isActive():
                    self.logger.info("Stopping timer")
                    self.timer.stop()
            
            # Cancel all tasks
            task_names = list(self.tasks.keys())
            if task_names:
                self.logger.info(f"Cancelling {len(task_names)} active tasks")
                for task_name in task_names:
                    try:
                        self._cancel_task(task_name)
                    except Exception as e:
                        self.logger.error(f"Error cancelling task {task_name}: {e}")
            
            # Process any pending events
            import time
            from PyQt6.QtCore import QCoreApplication
            QCoreApplication.processEvents()
            
            # Allow some time for tasks to finish
            time.sleep(0.2)  # Slightly longer delay to allow cancellation to take effect
            
            # Process events again
            QCoreApplication.processEvents()
            
            # Stop the event loop
            if self.loop and not self.loop.is_closed():
                try:
                    self.logger.info("Stopping event loop")
                    
                    # First, stop the loop
                    if self.loop.is_running():
                        self.loop.stop()
                    
                    # Run final iteration to handle task cancellation
                    for _ in range(10):  # Run more iterations to ensure all callbacks are processed
                        if not self.loop.is_closed():
                            # Process any pending callbacks in the loop
                            self.loop.call_soon_threadsafe(lambda: None)  # Dummy callback to wake the loop
                            self.loop.stop()
                            self.loop.run_forever()
                        else:
                            break
                    
                    # Process Qt events one more time
                    QCoreApplication.processEvents()
                    
                    # Close the loop
                    if not self.loop.is_closed():
                        try:
                            # Use run_until_complete with a completed future to flush any callbacks
                            future = asyncio.Future(loop=self.loop)
                            future.set_result(None)
                            self.loop.run_until_complete(future)
                        except Exception as e:
                            self.logger.error(f"Error flushing event loop: {e}")
                        
                        # Finally close the loop
                        self.loop.close()
                        
                    self.loop = None
                except Exception as e:
                    self.logger.error(f"Error stopping event loop: {e}")
            
            # Additional cleanup for PyQt integration
            self.tasks.clear()
            
            # Final processing of Qt events
            QCoreApplication.processEvents()
            
            self.logger.info("AsyncTaskManager cleanup completed")
                
        except Exception as e:
            self.logger.error(f"Error during AsyncTaskManager cleanup: {e}")
            # Try to ensure critical cleanup happens even with errors
            try:
                if hasattr(self, 'timer') and self.timer.isActive():
                    self.timer.stop()
                
                if self.loop and not self.loop.is_closed():
                    self.loop.close()
                    
                self.tasks.clear()
            except:
                pass 