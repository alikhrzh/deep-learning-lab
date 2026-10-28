from fastapi.testclient import TestClient

from deep_learning_lab.api import app


client = TestClient(app)


def test_health_describes_reproducible_model():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model": "numpy-mlp",
        "training_seed": 11,
        "training_samples": 240,
    }


def test_predict_returns_probability_and_binary_label():
    response = client.post("/predict", json={"x1": 0.25, "x2": -0.4})
    assert response.status_code == 200
    prediction = response.json()
    assert 0.0 <= prediction["probability"] <= 1.0
    assert prediction["label"] in (0, 1)
    assert prediction["label"] == int(prediction["probability"] >= 0.5)


def test_predict_rejects_incomplete_points():
    assert client.post("/predict", json={"x1": 0.25}).status_code == 422
