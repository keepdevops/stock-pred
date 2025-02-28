import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TestDataPreparation(unittest.TestCase):
    def setUp(self):
        """Set up test data"""
        # Create sample data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(100, 200, 100),
            'low': np.random.uniform(100, 200, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 5000, 100),
            'ticker': ['BZ=F'] * 100,
            'sector': ['Energy'] * 100,
            'updated_at': [datetime.now()] * 100
        })

    def test_data_preparation_steps(self):
        """Test each step of data preparation"""
        print("\n=== Starting Data Preparation Test ===")
        
        try:
            # Step 1: Check if DataFrame is valid
            print("\nStep 1: Validating DataFrame")
            self.assertFalse(self.sample_data.empty, "DataFrame is empty")
            print("✓ DataFrame is valid and not empty")
            print(f"Initial columns: {self.sample_data.columns.tolist()}")

            # Step 2: Convert column names to lowercase
            print("\nStep 2: Converting column names to lowercase")
            self.sample_data.columns = self.sample_data.columns.str.lower()
            print(f"Columns after lowercase conversion: {self.sample_data.columns.tolist()}")

            # Step 3: Check for required columns
            print("\nStep 3: Checking required columns")
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in self.sample_data.columns]
            self.assertEqual(len(missing_columns), 0, f"Missing columns: {missing_columns}")
            print("✓ All required columns present")

            # Step 4: Set date as index
            print("\nStep 4: Setting date as index")
            if 'date' in self.sample_data.columns:
                self.sample_data.set_index('date', inplace=True)
                print("✓ Date set as index")
            else:
                raise ValueError("Date column not found")

            # Step 5: Calculate technical indicators
            print("\nStep 5: Calculating technical indicators")
            try:
                self.sample_data['ma20'] = self.sample_data['close'].rolling(window=20).mean()
                self.sample_data['rsi'] = self._calculate_rsi(self.sample_data['close'])
                self.sample_data['macd'] = self._calculate_macd(self.sample_data['close'])
                self.sample_data['ma50'] = self.sample_data['close'].rolling(window=50).mean()
                print("✓ Technical indicators calculated")
                print(f"Final columns: {self.sample_data.columns.tolist()}")
            except Exception as e:
                raise ValueError(f"Failed to calculate technical indicators: {str(e)}")

            # Step 6: Check for NaN values
            print("\nStep 6: Checking for NaN values")
            nan_counts = self.sample_data.isna().sum()
            print("NaN counts per column:")
            print(nan_counts)
            
            # Step 7: Prepare sequences
            print("\nStep 7: Preparing sequences")
            sequence_length = 10
            features = self.sample_data[['open', 'high', 'low', 'close', 'volume', 'ma20', 'rsi', 'macd', 'ma50']].values
            
            if len(features) < sequence_length:
                raise ValueError(f"Not enough data points. Need at least {sequence_length}, got {len(features)}")
                
            X, y = self._create_sequences(features, sequence_length)
            print(f"✓ Created sequences: X shape: {X.shape}, y shape: {y.shape}")

            # Step 8: Split data
            print("\nStep 8: Splitting data")
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            print(f"✓ Data split - Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

            return (X_train, y_train), (X_val, y_val)

        except Exception as e:
            print(f"\n❌ Error in data preparation: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            print("\nTraceback:")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices):
        """Calculate MACD"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        return exp1 - exp2

    def _create_sequences(self, data, sequence_length):
        """Create sequences for training"""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:(i + sequence_length)])
            y.append(data[i + sequence_length, 3])  # 3 is the index for 'close' price
        return np.array(X), np.array(y)

if __name__ == '__main__':
    unittest.main(verbosity=2) 