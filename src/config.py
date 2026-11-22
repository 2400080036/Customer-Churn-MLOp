"""Central settings: paths, feature definitions and allowed values."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
DATA_CARD_PATH = BASE_DIR / "data" / "data_card.json"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "churn_model.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"
REPORT_DIR = BASE_DIR / "reports"
LOG_DIR = BASE_DIR / "logs"
PREDICTION_LOG = LOG_DIR / "predictions.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

ID_COLUMN = "customer_id"
TARGET = "churn"          # "Yes" / "No" in the csv

NUMERIC = ["tenure", "monthly_charges"]
BINARY = ["senior_citizen"]   # 0 / 1

# categorical column -> allowed values (also used for validation and the web form)
CATEGORICAL = {
    "gender": ["Female", "Male"],
    "partner": ["No", "Yes"],
    "dependents": ["No", "Yes"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "online_security": ["No", "Yes"],
    "tech_support": ["No", "Yes"],
    "contract": ["Month-to-month", "One year", "Two year"],
    "paperless_billing": ["No", "Yes"],
    "payment_method": ["Electronic check", "Mailed check",
                       "Bank transfer (automatic)", "Credit card (automatic)"],
}

FEATURES = NUMERIC + BINARY + list(CATEGORICAL)

# risk bands on the churn probability
def risk_level(probability: float) -> str:
    if probability < 0.20:
        return "Low"
    if probability < 0.50:
        return "Medium"
    return "High"
