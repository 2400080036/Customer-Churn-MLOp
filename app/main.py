from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import joblib
import uvicorn
import webbrowser
import threading


# ==========================================
# Customer Churn Prediction Application
# ==========================================

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"
FRONTEND_PATH = BASE_DIR / "frontend" / "index.html"


# Create FastAPI application
app = FastAPI(
    title="Customer Churn Prediction",
    description="ML-based Customer Churn Prediction System",
    version="1.0"
)


# ==========================================
# Load trained ML model
# ==========================================

model = joblib.load(MODEL_PATH)


# ==========================================
# Input data structure
# ==========================================

class CustomerData(BaseModel):
    gender: int
    senior_citizen: int
    tenure: int
    monthly_charges: float
    contract: int
    internet_service: int


# ==========================================
# Home page
# ==========================================

@app.get("/")
def home():
    return FileResponse(FRONTEND_PATH)


# ==========================================
# Prediction API
# ==========================================

@app.post("/predict")
def predict(data: CustomerData):

    # Prepare customer information for the model
    features = [[
        data.gender,
        data.senior_citizen,
        data.tenure,
        data.monthly_charges,
        data.contract,
        data.internet_service
    ]]

    # Make prediction
    prediction = model.predict(features)[0]

    # Handle different model output formats
    if str(prediction).lower() in ["yes", "1", "true"]:
        prediction_value = 1
        result = "Customer is likely to churn"
    else:
        prediction_value = 0
        result = "Customer is unlikely to churn"

    return {
        "result": result,
        "prediction": prediction_value
    }


# ==========================================
# Run application from PyCharm
# ==========================================

if __name__ == "__main__":

    # Automatically open the Customer Churn application
    def open_browser():
        webbrowser.open("http://127.0.0.1:8000/")

    threading.Timer(1.5, open_browser).start()

    # Start FastAPI server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )