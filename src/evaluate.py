from pathlib import Path
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

project_path = Path(__file__).resolve().parent.parent

data = pd.read_csv(project_path / "data" / "customer_churn.csv")

data = data.drop("customer_id", axis=1)

data["gender"] = data["gender"].map({"Male": 1, "Female": 0})
data["contract"] = data["contract"].map({
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
})
data["internet_service"] = data["internet_service"].map({
    "DSL": 0,
    "Fiber optic": 1
})
data["churn"] = data["churn"].map({"No": 0, "Yes": 1})

X = data.drop("churn", axis=1)
y = data["churn"]

model = joblib.load(project_path / "models" / "churn_model.pkl")

predictions = model.predict(X)

print("Accuracy:", accuracy_score(y, predictions))

print("\nClassification Report:")
print(classification_report(y, predictions))