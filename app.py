from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
from models.neural_network import NeuralNetwork
from utils.data_preprocessing import preprocess_input

app = Flask(__name__)

def ensure_model_exists():
    if not os.path.exists('models/saved_models/trained_model.pkl'):
        from train_model import train_and_save_model
        train_and_save_model()

@app.before_first_request
def initialize():
    ensure_model_exists()

def load_model():
    model_path = 'models/saved_models/trained_model.pkl'
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        # Train a new model if file doesn't exist
        from train_model import train_and_save_model
        return train_and_save_model()

model = load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:        
        data = {
            'TV': float(request.form['tv']),
            'Radio': float(request.form['radio']),
            'Newspaper': float(request.form['newspaper'])
        }
                
        processed_input = preprocess_input(data)
                
        prediction = model.predict(processed_input)[0][0]
        
        return render_template('predict.html', 
                             prediction=f'${prediction:,.2f}',
                             input_data=data)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)