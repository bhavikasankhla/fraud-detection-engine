from fastapi.testclient import TestClient
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_legitimate():
    payload = {
        "TransactionAmt": 25.00,
        "TransactionDT": 86400,
        "card1": 1234,
        "P_emaildomain": "outlook.com"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert "decision" in data
    assert data["decision"] in ["FRAUD", "LEGITIMATE"]
    assert 0.0 <= data["fraud_probability"] <= 1.0


def test_predict_high_risk():
    payload = {
        "TransactionAmt": 2500.00,
        "TransactionDT": 3600,
        "card1": 9500,
        "P_emaildomain": "gmail.com"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_level"] in ["HIGH", "MEDIUM", "LOW"]


def test_invalid_transaction_amount():
    payload = {
        "TransactionAmt": -50.00,
        "TransactionDT": 3600
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error expected