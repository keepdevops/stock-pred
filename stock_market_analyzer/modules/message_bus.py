import logging
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Message:
    """Represents a message in the message bus system."""
    type: str
    data: Any
    timestamp: datetime
    source: str

class MessageBus:
    """Message bus for inter-tab communication."""
    
    def __init__(self):
        """Initialize the message bus."""
        self.subscribers: Dict[str, List[Callable]] = {}
        self.logger = logging.getLogger(__name__)
        self.message_history: List[Message] = []
        
    def subscribe(self, topic: str, callback: Callable):
        """Subscribe to a topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        if callback not in self.subscribers[topic]:
            self.subscribers[topic].append(callback)
            self.logger.debug(f"Added subscriber to topic '{topic}': {callback.__name__}")
        
    def publish(self, topic: str, message_type: str, data: dict):
        """Publish a message to subscribers."""
        self.logger.debug(f"Publishing message - Topic: {topic}, Type: {message_type}, Data: {data}")
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    callback(topic, message_type, data)
                    self.logger.debug(f"Successfully delivered message to {callback.__name__}")
                except Exception as e:
                    self.logger.error(f"Error delivering message to {callback.__name__}: {e}")
                    
    def get_message_history(self, message_type: str = None) -> List[Message]:
        """
        Get message history, optionally filtered by message type.
        
        Args:
            message_type: Optional message type to filter by
            
        Returns:
            List of messages matching the criteria
        """
        if message_type:
            return [msg for msg in self.message_history if msg.type == message_type]
        return self.message_history
        
    def clear_history(self):
        """Clear the message history."""
        self.message_history.clear()
        self.logger.debug("Message history cleared")
        
    def get_subscribers(self, message_type: str = None) -> Dict[str, List[Callable]]:
        """
        Get subscribers, optionally filtered by message type.
        
        Args:
            message_type: Optional message type to filter by
            
        Returns:
            Dictionary of subscribers
        """
        if message_type:
            return {message_type: self.subscribers.get(message_type, [])}
        return self.subscribers

    def unsubscribe(self, topic: str, callback: Callable):
        """Unsubscribe from a topic."""
        if topic in self.subscribers and callback in self.subscribers[topic]:
            self.subscribers[topic].remove(callback)
            self.logger.debug(f"Removed subscriber from topic '{topic}': {callback.__name__}")
            if not self.subscribers[topic]:
                del self.subscribers[topic] 