# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('turbine_model.pkl')  # Ensure you have the trained model file
scaler = joblib.load('scaler.pkl')        # Ensure you have the scaler file

@app.route('/')
def home():
    return "Turbine Failure Prediction API"

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the data sent as JSON in the POST request
            data = request.get_json(force=True)

            # Convert the JSON data into a DataFrame
            input_data = pd.DataFrame([data])

            # Scale the data using the scaler
            input_data_scaled = scaler.transform(input_data)

            # Predict using the loaded model
            prediction = model.predict(input_data_scaled)

            # Return the prediction as JSON
            result = {'prediction': 'Failure' if prediction[0] == 1 else 'No Failure'}
            return jsonify(result)

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    # For GET requests (for testing only)
    else:
        return jsonify({"message": "Use POST method to get predictions."})


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
