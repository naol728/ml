from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load the trained model (Make sure the path to 'diabetes_model.pkl' is correct)
model = joblib.load('diabetes_model.pkl')

class InputData(BaseModel):
    pregnancies: int
    glucose: int
    blood_pressure: int
    skin_thickness: int
    insulin: int
    bmi: float
    diabetes_pedigree_function: float
    age: int

@app.post("/predict")
def predict(data: InputData):
    # Convert the input data into a numpy array for prediction
    input_data = np.array([[data.pregnancies, data.glucose, data.blood_pressure, data.skin_thickness,
                            data.insulin, data.bmi, data.diabetes_pedigree_function, data.age]])

    # Make the prediction
    prediction = model.predict(input_data)
    
    # Return the prediction as a JSON response
    return {"prediction": int(prediction[0])}  # Ensure prediction is an integer (0 or 1)
