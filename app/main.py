from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="Churn Prediction API")

# Load model on startup
try:
    model = joblib.load("models/churn_model.pkl")
    features = joblib.load("models/feature_names.pkl")
except FileNotFoundError:
    model = None
    features = None

class CustomerInput(BaseModel):
    CustomerID: int
    Recency: int
    Frequency: int
    Monetary: float

@app.get("/")
def read_root():
    return {"status": "API is running. Go to /docs for Swagger UI"}

@app.post("/predict")
def predict_churn(customer: CustomerInput):
    if model is None:
        return {"error": "Model not found. Please run training first."}
    
    input_data = pd.DataFrame([{
        "Recency": customer.Recency,
        "Frequency": customer.Frequency,
        "Monetary": customer.Monetary
    }])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    return {
        "CustomerID": customer.CustomerID,
        "Churn": bool(prediction),
        "Churn_Probability": float(probability),
        "Risk_Level": "High" if probability > 0.5 else "Low"
    }