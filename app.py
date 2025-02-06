from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import warnings

warnings.simplefilter("ignore")

app = Flask(__name__)
CORS(app)  # Enable CORS for the app

# Load the trained model
model = joblib.load("diabetes_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Diabetes Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.json

        # Convert input data into a NumPy array
        input_data = np.array([[data["pregnancies"], data["glucose"], data["blood_pressure"],
                                data["skin_thickness"], data["insulin"], data["bmi"],
                                data["diabetes_pedigree_function"], data["age"]]])

        # Make prediction
        prediction = model.predict(input_data)

        # Return response
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
