import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        """Load the advertising dataset"""
        df = pd.read_csv(filepath)
        df.columns = ['TV', 'Radio', 'Newspaper', 'Sales']
        return df
    
    def preprocess_data(self, df):
        """Preprocess the data for training"""
        X = df[['TV', 'Radio', 'Newspaper']]
        y = df['Sales']
        
        X_scaled = self.scaler.fit_transform(X)
        
        with open('models/saved_models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        return X_scaled, y.values.reshape(-1, 1)

def preprocess_input(data):
    """Preprocess new input data"""
    with open('models/saved_models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    input_array = np.array([[
        data['TV'],
        data['Radio'],
        data['Newspaper']
    ]])
    
    scaled_input = scaler.transform(input_array)
    
    return scaled_input