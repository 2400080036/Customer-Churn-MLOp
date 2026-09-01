from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Customer Churn Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:63342",
        "http://127.0.0.1:63342"
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)



model_path = Path(__file__).resolve().parent.parent / "models" / "churn_model.pkl"
model = joblib.load(model_path)


class Customer(BaseModel):
    gender: int
    senior_citizen: int
    tenure: int
    monthly_charges: float
    contract: int
    internet_service: int


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running"}


@app.post("/predict")
def predict(customer: Customer):
    data = pd.DataFrame([customer.model_dump()])

    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "Customer will Churn"
    else:
        result = "Customer will Not Churn"

    return {
        "prediction": int(prediction),
        "result": result
    }