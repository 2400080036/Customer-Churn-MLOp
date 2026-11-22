import os

os.environ["CHURN_DISABLE_LOG"] = "1"      # tests must not write to the real prediction log

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

CUSTOMER = {
    "gender": "Male", "senior_citizen": 0, "partner": "No", "dependents": "No",
    "tenure": 3, "monthly_charges": 95.0, "internet_service": "Fiber optic",
    "online_security": "No", "tech_support": "No", "contract": "Month-to-month",
    "paperless_billing": "Yes", "payment_method": "Electronic check",
}
LOYAL = {**CUSTOMER, "tenure": 60, "monthly_charges": 45.0, "internet_service": "DSL",
         "online_security": "Yes", "tech_support": "Yes", "contract": "Two year",
         "paperless_billing": "No", "payment_method": "Credit card (automatic)"}


def test_home():
    assert client.get("/").status_code == 200


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_model_info_reports_test_metrics():
    info = client.get("/model-info").json()
    assert info["rows"] >= 500
    assert 0.5 < info["test_metrics"]["roc_auc"] <= 1


def test_predict_returns_probability_and_factors():
    body = client.post("/predict", json=CUSTOMER).json()
    assert 0 <= body["probability"] <= 1
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert len(body["factors"]) == 12


def test_risky_customer_scores_higher_than_loyal_customer():
    risky = client.post("/predict", json=CUSTOMER).json()["probability"]
    loyal = client.post("/predict", json=LOYAL).json()["probability"]
    assert risky > loyal + 0.3


def test_gender_does_not_change_the_prediction_much():
    male = client.post("/predict", json=CUSTOMER).json()["probability"]
    female = client.post("/predict", json={**CUSTOMER, "gender": "Female"}).json()["probability"]
    assert abs(male - female) < 0.05


def test_invalid_input_is_rejected():
    assert client.post("/predict", json={**CUSTOMER, "contract": "Five year"}).status_code == 422
    assert client.post("/predict", json={**CUSTOMER, "tenure": -4}).status_code == 422


def test_monitoring_endpoint():
    assert "status" in client.get("/monitoring").json()
