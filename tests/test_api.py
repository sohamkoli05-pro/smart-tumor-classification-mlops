
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Smart Tumor Classification API is running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_wrong_number_of_features():
    response = client.post("/predict", json={"features": [0]})
    assert response.status_code == 400
    assert "Expected 30 features" in response.json()["detail"]


def test_predict_valid_features():
    features = [0.0] * 30

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "class_name" in response.json()
    assert "notice" in response.json()
