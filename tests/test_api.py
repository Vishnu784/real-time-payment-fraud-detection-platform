from fastapi.testclient import TestClient

from app.main import app


def sample_transaction(transaction_id: str = "txn-001") -> dict:
    return {
        "transaction_id": transaction_id,
        "amount": 8250.0,
        "hour": 2,
        "distance_from_home_km": 480.0,
        "merchant_risk_score": 0.91,
        "velocity_1h": 12,
        "failed_attempts_24h": 5,
        "is_foreign": True,
        "card_present": False,
        "account_age_days": 15,
    }


def test_health_and_prediction():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"

        response = client.post("/predict", json=sample_transaction())
        assert response.status_code == 201
        body = response.json()
        assert 0 <= body["fraud_probability"] <= 1
        assert body["is_fraud"] is True
        assert len(body["risk_factors"]) == 3


def test_duplicate_transaction_is_rejected():
    with TestClient(app) as client:
        payload = sample_transaction("txn-duplicate")
        assert client.post("/predict", json=payload).status_code == 201
        assert client.post("/predict", json=payload).status_code == 409


def test_validation_and_metrics():
    with TestClient(app) as client:
        invalid = sample_transaction("txn-invalid")
        invalid["hour"] = 25
        assert client.post("/predict", json=invalid).status_code == 422

        metrics = client.get("/metrics")
        assert metrics.status_code == 200
        assert metrics.json()["total_scored"] >= 1

