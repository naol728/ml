from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load the trained model
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
    input_data = np.array([[data.pregnancies, data.glucose, data.blood_pressure, data.skin_thickness,
                            data.insulin, data.bmi, data.diabetes_pedigree_function, data.age]])
    
    prediction = model.predict(input_data)
    return {"prediction": prediction[0]}
