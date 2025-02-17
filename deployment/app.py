from flask import Flask, request, jsonify
import pickle
import os
import numpy as np
import logging

def load_model():
    """Load the trained stacked model with exception handling."""
    model_path = 'models/stacked_model.pkl'
    if not os.path.exists(model_path):
        logging.error(f"Model file not found at {model_path}. Please ensure the model is trained and saved correctly.")
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data['model'], model_data.get('metadata', {})
    except (pickle.UnpicklingError, KeyError, EOFError) as e:
        logging.error(f"Error loading the model: {e}")
        raise RuntimeError("Failed to load model due to corruption or missing keys.")

app = Flask(__name__)
try:
    model, metadata = load_model()
except Exception as e:
    logging.error(f"Application failed to start due to model loading issue: {e}")
    model, metadata = None, {}

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to make predictions using the stacked model."""
    if model is None:
        return jsonify({'error': "Model is not available. Please check logs for details."}), 500
    
    try:
        data = request.get_json()
        if 'features' not in data or not isinstance(data['features'], list):
            return jsonify({'error': "Invalid input. 'features' must be a list."}), 400
        
        features = np.array(data['features']).reshape(1, -1)
        prediction = model.predict(features).tolist()
        model_version = metadata.get('model_version', 'unknown')
        return jsonify({'prediction': prediction, 'model_version': model_version})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/metadata', methods=['GET'])
def get_metadata():
    """API endpoint to retrieve model metadata."""
    return jsonify(metadata)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
