"""Loads the saved model and turns one customer record into a prediction."""
import json
import os

import joblib
import pandas as pd

from src.config import FEATURES, METADATA_PATH, MODEL_PATH, risk_level
from src.monitoring import log_prediction

FRIENDLY = {
    "gender": "Gender", "senior_citizen": "Senior citizen", "partner": "Partner",
    "dependents": "Dependents", "tenure": "Tenure", "monthly_charges": "Monthly charges",
    "internet_service": "Internet service", "online_security": "Online security",
    "tech_support": "Tech support", "contract": "Contract",
    "paperless_billing": "Paperless billing", "payment_method": "Payment method",
}


class Predictor:
    def __init__(self, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Train the model first: python src/train.py")
        self.model = joblib.load(model_path)
        self.metadata = json.loads(metadata_path.read_text())
        self.threshold = self.metadata["threshold"]
        self.reference = self.metadata["reference_customer"]

    def predict(self, record: dict, log: bool = True) -> dict:
        row = {f: record[f] for f in FEATURES}
        # one batch: the customer + the same customer with ONE feature set to the typical value
        frame = pd.DataFrame([row] + [{**row, f: self.reference[f]} for f in FEATURES])
        proba = self.model.predict_proba(frame)[:, 1]
        p = float(proba[0])

        factors = [{"feature": f, "label": FRIENDLY[f], "value": row[f],
                    "effect": float(p - proba[i + 1])}   # + raises risk vs a typical customer
                   for i, f in enumerate(FEATURES)]
        factors.sort(key=lambda d: abs(d["effect"]), reverse=True)

        if log and os.environ.get("CHURN_DISABLE_LOG") != "1":
            log_prediction(row, p, self.metadata["model_version"])

        likely = p >= self.threshold
        return {
            "result": "Customer is likely to churn" if likely else "Customer is unlikely to churn",
            "prediction": int(likely),
            "probability": p,
            "risk_level": risk_level(p),
            "threshold": self.threshold,
            "factors": factors,
            "model_version": self.metadata["model_version"],
        }
