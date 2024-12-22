import unittest
import numpy as np
import pandas as pd
import os
from utils.data_preprocessing import DataPreprocessor, preprocess_input

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        self.preprocessor = DataPreprocessor()
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'TV': [100, 200, 300],
            'Radio': [20, 40, 60],
            'Newspaper': [10, 20, 30],
            'Sales': [22, 45, 67]
        })
        
        # Save sample data to temp file
        self.temp_file = 'temp_test_data.csv'
        self.sample_data.to_csv(self.temp_file, index=False)

    def tearDown(self):
        # Clean up temp file
        if os.path.exists(self.temp_file):
            os.remove(self.temp_file)

    def test_load_data(self):
        df = self.preprocessor.load_data(self.temp_file)
        self.assertTrue(isinstance(df, pd.DataFrame))
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), ['TV', 'Radio', 'Newspaper', 'Sales'])

    def test_preprocess_data(self):
        X_scaled, y = self.preprocessor.preprocess_data(self.sample_data)
        
        # Check shapes
        self.assertEqual(X_scaled.shape, (3, 3))
        self.assertEqual(y.shape, (3, 1))
        
        # Check scaling
        self.assertAlmostEqual(X_scaled.mean(), 0, places=10)
        self.assertAlmostEqual(X_scaled.std(), 1, places=10)

    def test_preprocess_input(self):
        # Test single input preprocessing
        input_data = {
            'TV': 150,
            'Radio': 30,
            'Newspaper': 15
        }
        
        # First process some data to save the scaler
        self.preprocessor.preprocess_data(self.sample_data)
        
        # Then test input preprocessing
        processed_input = preprocess_input(input_data)
        
        self.assertEqual(processed_input.shape, (1, 3))

if __name__ == '__main__':
    unittest.main()