"""Convert the public 'Telco Customer Churn' csv (Kaggle / IBM) into this project's format.

Usage:
    python src/prepare_telco.py path/to/WA_Fn-UseC_-Telco-Customer-Churn.csv
Then run:  python src/train.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import DATA_CARD_PATH, DATA_PATH
from src.data import validate

RENAME = {
    "customerID": "customer_id", "gender": "gender", "SeniorCitizen": "senior_citizen",
    "Partner": "partner", "Dependents": "dependents", "tenure": "tenure",
    "InternetService": "internet_service", "OnlineSecurity": "online_security",
    "TechSupport": "tech_support", "Contract": "contract",
    "PaperlessBilling": "paperless_billing", "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges", "Churn": "churn",
}

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    raw = pd.read_csv(sys.argv[1])
    df = raw[list(RENAME)].rename(columns=RENAME)
    for col in ["online_security", "tech_support"]:
        df[col] = df[col].replace({"No internet service": "No"})
    df["monthly_charges"] = pd.to_numeric(df["monthly_charges"], errors="coerce")
    validate(df)
    df.to_csv(DATA_PATH, index=False)
    DATA_CARD_PATH.write_text(json.dumps({
        "source": "IBM Telco Customer Churn (public dataset)", "is_real": True, "rows": len(df)}, indent=2))
    print(f"Saved {len(df)} rows to {DATA_PATH}. Now run: python src/train.py")
