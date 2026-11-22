
from pathlib import Path
import sys
import logging
import webbrowser
import threading
import json
import pandas as pd

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn


# =====================================================
# PROJECT PATHS
# =====================================================

APP_DIR = Path(__file__).resolve().parent
BASE_DIR = APP_DIR.parent
FRONTEND_DIR = BASE_DIR / "frontend"

INDEX_PATH = FRONTEND_DIR / "index.html"
CUSTOMER_PATH = FRONTEND_DIR / "customer.html"
RESULT_PATH = FRONTEND_DIR / "result.html"

DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
METADATA_PATH = BASE_DIR / "models" / "metadata.json"
PREDICTION_LOG_PATH = BASE_DIR / "logs" / "predictions.csv"

sys.path.insert(0, str(BASE_DIR))


# =====================================================
# LOGGING
# =====================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =====================================================
# MACHINE LEARNING IMPORTS
# =====================================================

predictor = None
drift_report = None
MODEL_STATUS = "not loaded"
IMPORT_ERROR = None

try:
    from src.predictor import Predictor
    from src.monitoring import drift_report

    predictor = Predictor()
    MODEL_STATUS = "loaded"

except Exception as error:
    MODEL_STATUS = "error"
    IMPORT_ERROR = str(error)
    logger.exception("Model loading failed")


# =====================================================
# LOAD MODEL METADATA
# =====================================================

def load_metadata():
    if not METADATA_PATH.is_file():
        logger.warning("Model metadata file not found")
        return {}

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        logger.exception("Unable to load model metadata")
        return {}


def get_row_count(metadata):
    # Prefer the actual dataset row count.
    if DATA_PATH.is_file():
        try:
            return int(len(pd.read_csv(DATA_PATH)))
        except Exception:
            logger.exception("Unable to read training dataset")

    # Use metadata only if the dataset is unavailable.
    for key in ("rows", "n_rows", "total_rows", "train_rows"):
        if key in metadata:
            try:
                return int(metadata[key])
            except (TypeError, ValueError):
                pass

    return 0


# =====================================================
# FASTAPI APPLICATION
# =====================================================

app = FastAPI(
    title="Customer Churn Prediction System",
    description="Machine Learning Based Customer Churn Prediction",
    version="1.0.0"
)


# =====================================================
# CUSTOMER INPUT MODEL
# =====================================================

class CustomerData(BaseModel):

    gender: Literal["Female", "Male"]
    senior_citizen: Literal[0, 1]
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]

    tenure: int = Field(ge=0)
    monthly_charges: float = Field(ge=0)

    internet_service: Literal[
        "DSL",
        "Fiber optic",
        "No"
    ]

    contract: Literal[
        "Month-to-month",
        "One year",
        "Two year"
    ]

    online_security: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    tech_support: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    paperless_billing: Literal[
        "Yes",
        "No"
    ]

    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]


# =====================================================
# FRONTEND FILE CHECK
# =====================================================

def serve_page(file_path):

    if not file_path.is_file():
        logger.error("Frontend file missing: %s", file_path)

        raise HTTPException(
            status_code=404,
            detail={
                "error": f"{file_path.name} not found",
                "expected_path": str(file_path),
                "frontend_folder": str(FRONTEND_DIR)
            }
        )

    return FileResponse(
        path=str(file_path),
        media_type="text/html"
    )


# =====================================================
# FRONTEND ROUTES
# =====================================================

@app.get("/")
def home():
    return serve_page(INDEX_PATH)


@app.get("/customer")
def customer_page():
    return serve_page(CUSTOMER_PATH)


@app.get("/result")
def result_page():
    return serve_page(RESULT_PATH)


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health_check():

    return {
        "status": "ok",
        "model_status": MODEL_STATUS,
        "frontend_files": {
            "index.html": INDEX_PATH.is_file(),
            "customer.html": CUSTOMER_PATH.is_file(),
            "result.html": RESULT_PATH.is_file()
        }
    }


# =====================================================
# MODEL INFORMATION
# =====================================================

@app.get("/model-info")
def model_info():

    metadata = load_metadata()

    return {
        "project": "Customer Churn Prediction System",
        "model_status": MODEL_STATUS,
        "model_loaded": predictor is not None,
        "model_error": IMPORT_ERROR,
        "rows": get_row_count(metadata),
        "test_metrics": metadata.get("test_metrics", {})
    }


# =====================================================
# MONITORING
# =====================================================

@app.get("/monitoring")
def monitoring():

    if drift_report is None:
        raise HTTPException(
            status_code=503,
            detail="Monitoring module unavailable"
        )

    try:
        metadata = load_metadata()

        baseline = metadata.get("drift_baseline")

        if baseline is None:
            baseline = metadata.get("baseline")

        if not baseline:
            return {
                "status": "NOT_CONFIGURED",
                "message": "Drift baseline is not available in model metadata."
            }

        if PREDICTION_LOG_PATH.is_file():
            recent = pd.read_csv(PREDICTION_LOG_PATH)
        else:
            recent = pd.DataFrame()

        report = drift_report(
            baseline=baseline,
            recent=recent
        )

        return {
            "status": "success",
            "monitoring": report
        }

    except Exception as error:
        logger.exception("Monitoring failed")

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# =====================================================
# API INFORMATION
# =====================================================

@app.get("/api")
def api_information():

    return {
        "application": "Customer Churn Prediction System",
        "endpoints": {
            "home": "/",
            "customer": "/customer",
            "result": "/result",
            "health": "/health",
            "model_info": "/model-info",
            "monitoring": "/monitoring",
            "prediction": "/predict",
            "docs": "/docs"
        }
    }


# =====================================================
# CUSTOMER CHURN PREDICTION
# =====================================================

@app.post("/predict")
def predict_churn(customer: CustomerData):

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {IMPORT_ERROR}"
        )

    try:
        customer_data = customer.model_dump()
        result = predictor.predict(customer_data)

        return result

    except Exception as error:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )


# =====================================================
# AUTOMATIC BROWSER OPENING
# =====================================================

def open_browser():
    webbrowser.open("http://127.0.0.1:8000/")


# =====================================================
# START APPLICATION
# =====================================================

if __name__ == "__main__":

    print("\n" + "=" * 55)
    print("CUSTOMER CHURN PREDICTION SYSTEM")
    print("=" * 55)

    print("\nPROJECT DIRECTORY:")
    print(BASE_DIR)

    print("\nFRONTEND DIRECTORY:")
    print(FRONTEND_DIR)

    print("\nMODEL STATUS:", MODEL_STATUS)

    if IMPORT_ERROR:
        print("MODEL ERROR:", IMPORT_ERROR)

    print("\nAPPLICATION URLS:")
    print("Welcome:  http://127.0.0.1:8000/")
    print("Customer: http://127.0.0.1:8000/customer")
    print("Result:   http://127.0.0.1:8000/result")
    print("Health:   http://127.0.0.1:8000/health")
    print("API Docs: http://127.0.0.1:8000/docs")

    print("\n" + "=" * 55)

    threading.Timer(2.0, open_browser).start()

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False
    )
