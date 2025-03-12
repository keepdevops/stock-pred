"""
Event bus for communication between components
"""
class EventBus:
    def __init__(self):
        self._subscribers = {}
        
    def subscribe(self, event_type, callback):
        """Subscribe to an event type"""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
        
    def unsubscribe(self, event_type, callback):
        """Unsubscribe from an event type"""
        if event_type in self._subscribers and callback in self._subscribers[event_type]:
            self._subscribers[event_type].remove(callback)
            
    def publish(self, event_type, data=None):
        """Publish an event to all subscribers"""
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                callback(data)
                
# Create a global event bus instance
event_bus = EventBus() 