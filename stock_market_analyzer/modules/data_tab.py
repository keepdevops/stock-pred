class DataTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.message_bus = MessageBus()
        self.message_bus.subscribe("heartbeat", self.handle_message)

    def setup_ui(self):
        # Implementation of setup_ui method
        pass

    def handle_message(self, message):
        # Implementation of handle_message method
        pass 