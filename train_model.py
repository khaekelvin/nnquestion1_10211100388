import pandas as pd
import numpy as np
import pickle
import os
from models.neural_network import NeuralNetwork
from utils.data_preprocessing import DataPreprocessor

def train_and_save_model():
    # Ensure directories exist
    os.makedirs('models/saved_models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    data = preprocessor.load_data('data/raw/AdvertisingBudgetandSales.csv')
    X_scaled, y = preprocessor.preprocess_data(data)
    
    # Initialize neural network
    input_size = X_scaled.shape[1]  # 3 features
    hidden_size = 8
    output_size = 1
    learning_rate = 0.01
    
    model = NeuralNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        learning_rate=learning_rate
    )
    
    # Train the model
    print("Training model...")
    losses = model.train(X_scaled, y, epochs=1000)
    
    # Save the trained model
    print("Saving model...")
    with open('models/saved_models/trained_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print("Model trained and saved successfully!")
    return model

if __name__ == "__main__":
    train_and_save_model()