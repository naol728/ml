from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load('diabetes_model.pkl')

# Initialize the Flask application
app = Flask(__name__)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get JSON data from the request
    data = request.get_json()

    # Ensure that the incoming data is in the correct format
    features = np.array([[
        data['Pregnancies'],
        data['Glucose'],
        data['BloodPressure'],
        data['SkinThickness'],
        data['Insulin'],
        data['BMI'],
        data['DiabetesPedigreeFunction'],
        data['Age']
    ]])

    # Make the prediction
    prediction = model.predict(features)

    # Return the result as JSON
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    # Run the Flask app on port 5000
    app.run(debug=True, port=5000)
