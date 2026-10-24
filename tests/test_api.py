from http import client

import requests
import os

# Адрес, на котором запущен ваш uvicorn (из скриншота — 8001)
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001")


def test_health():
    response = client.get(f"{BASE_URL}/healthcheck")  # было /health
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_model_info():
    response = requests.get(f"{BASE_URL}/model-info")
    assert response.status_code == 200
    data = response.json()
    if "model_type" in data:
        assert data["model_type"] == "RandomForestRegressor"
    else:
        assert "error" in data


def test_predict():
    payload = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.7,
        "citric_acid": 0.0,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    prediction = response.json()
    assert isinstance(prediction, (int, float))
    assert 3.0 <= prediction <= 9.0


def test_predict_invalid_data():
    payload = {"fixed_acidity": 7.4}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 422
