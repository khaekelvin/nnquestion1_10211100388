import pickle
import numpy as np
from utils.data_preprocessing import preprocess_input

def test_model():    
    with open('models/saved_models/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
        
    test_input = {
        'TV': 230.1,
        'Radio': 37.8,
        'Newspaper': 69.2
    }
        
    processed_input = preprocess_input(test_input)
    prediction = model.predict(processed_input)[0][0]
    
    print(f"Test Input: {test_input}")
    print(f"Predicted Sales: ${prediction:,.2f}")

if __name__ == "__main__":
    test_model()