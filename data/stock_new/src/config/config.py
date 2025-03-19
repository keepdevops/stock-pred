class DataCollectionConfig:
    def __init__(self):
        self.data_collection = {
            'historical': {
                'start_date': '2010-01-01',
                'end_date': 'now',
                'interval': '1d'
            },
            'realtime': {
                'interval': '1m',
                'period': '1d'
            }
        } 