from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
import os
from models.neural_network import NeuralNetwork
from utils.data_preprocessing import preprocess_input

app = Flask(__name__)

# Load the trained model and scaler
model_path = 'models/saved_models/trained_model.pkl'
if os.path.getsize(model_path) > 0:  # Check if the file is not empty
    with open(model_path, 'rb') as f:
        try:
            model = pickle.load(f)
        except EOFError:
            print("Error: The model file is empty or corrupted.")
else:
    print("Error: The model file is empty.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        data = {
            'TV': float(request.form['tv']),
            'Radio': float(request.form['radio']),
            'Newspaper': float(request.form['newspaper'])
        }
        
        # Preprocess input
        processed_input = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(processed_input)[0][0]
        
        return render_template('predict.html', 
                             prediction=f'${prediction:,.2f}',
                             input_data=data)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)