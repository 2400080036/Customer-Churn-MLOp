from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

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

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

joblib.dump(model, project_path / "models" / "churn_model.pkl")

print("Model saved successfully!")