# app/api.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
import sys

# Add project root to Python path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.crop_predictor import load_models, predict_crop

# -----------------------------
# FastAPI setup
# -----------------------------
app = FastAPI(title="🌾 Sustainable Crop Recommendation API")

# Load models + encoders
MODELS, ENCODERS = load_models()

# -----------------------------
# Input schema
# -----------------------------
class FarmData(BaseModel):
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float

# -----------------------------
# API Endpoints
# -----------------------------
@app.get("/")
def home():
    return {"message": "Welcome to the Sustainable Crop Recommendation API 🌍"}

@app.post("/predict")
def predict_crop_api(data: FarmData):
    input_df = pd.DataFrame([data.dict()])
    predictions = predict_crop(input_df, MODELS, ENCODERS)
    return {"predictions": predictions}
