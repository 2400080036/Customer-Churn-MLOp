from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_predict():
    data = {
        "gender": 1,
        "senior_citizen": 0,
        "tenure": 12,
        "monthly_charges": 70.5,
        "contract": 0,
        "internet_service": 1
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "result" in response.json()