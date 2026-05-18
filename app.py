from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load trained model
model = joblib.load("model.pkl")

# Create FastAPI app
app = FastAPI()

# Input schema
class InputData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Root endpoint
@app.get("/")
def home():
    return {"message": "ML Model API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):

    input_data = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(input_data)

    return {
        "prediction": int(prediction[0])
    }