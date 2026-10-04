from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import joblib
import uvicorn
import webbrowser
import threading


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

FRONTEND_DIR = BASE_DIR / "frontend"

MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

INDEX_PATH = FRONTEND_DIR / "index.html"
PREDICTION_PATH = FRONTEND_DIR / "prediction.html"
RESULT_PATH = FRONTEND_DIR / "result.html"


# ==========================================
# FASTAPI APPLICATION
# ==========================================

app = FastAPI(
    title="Customer Churn Prediction",
    description="ML-based Customer Churn Prediction System",
    version="1.0"
)


# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load(MODEL_PATH)


# ==========================================
# INPUT DATA
# ==========================================

class CustomerData(BaseModel):

    gender: int
    senior_citizen: int
    tenure: int
    monthly_charges: float
    contract: int
    internet_service: int


# ==========================================
# PAGE 1 - WELCOME
# ==========================================

@app.get("/")
def home():

    return FileResponse(
        INDEX_PATH
    )


# ==========================================
# PAGE 2 - CUSTOMER INFORMATION
# ==========================================

@app.get("/prediction")
@app.get("/prediction.html")
def prediction_page():

    return FileResponse(
        PREDICTION_PATH
    )


# ==========================================
# PAGE 3 - RESULT
# ==========================================

@app.get("/result")
@app.get("/result.html")
def result_page():

    return FileResponse(
        RESULT_PATH
    )


# ==========================================
# PREDICTION API
# ==========================================

@app.post("/predict")
def predict(data: CustomerData):

    features = [[
        data.gender,
        data.senior_citizen,
        data.tenure,
        data.monthly_charges,
        data.contract,
        data.internet_service
    ]]


    # MODEL PREDICTION

    prediction = model.predict(features)[0]


    # CONVERT PREDICTION

    if str(prediction).lower() in [
        "yes",
        "1",
        "true",
        "churn"
    ]:

        prediction_value = 1

        result = "Customer is likely to churn"

    else:

        prediction_value = 0

        result = "Customer is unlikely to churn"


    # ======================================
    # PROBABILITY
    # ======================================

    probability = None

    try:

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                features
            )[0]


            if hasattr(model, "classes_"):

                classes = list(
                    model.classes_
                )


                if 1 in classes:

                    churn_index = classes.index(1)

                    probability = float(
                        probabilities[churn_index]
                    )


                elif "Yes" in classes:

                    churn_index = classes.index("Yes")

                    probability = float(
                        probabilities[churn_index]
                    )

    except Exception:

        probability = None


    # ======================================
    # RESPONSE
    # ======================================

    response = {

        "result": result,

        "prediction": prediction_value

    }


    if probability is not None:

        response["probability"] = probability


    return response


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health_check():

    return {
        "status": "online",
        "message": "Customer Churn Prediction API is running"
    }


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":


    def open_browser():

        webbrowser.open(
            "http://127.0.0.1:8000/"
        )


    threading.Timer(
        1.5,
        open_browser
    ).start()


    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )