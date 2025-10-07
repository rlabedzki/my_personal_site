import pytest
import httpx

BASE_URL = "http://127.0.0.1:8000"  # kontener Titanic w docker-compose

# Testy poprawnych danych
@pytest.mark.parametrize(
    "payload, expected_prediction",
    [
        ({"Pclass": 3, "Sex": "male", "Age": 22, "Fare": 7.25}, 0),
        ({"Pclass": 1, "Sex": "female", "Age": 38, "Fare": 71.28}, 1),
    ]
)

def test_predict_valid(payload, expected_prediction):
    response = httpx.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    # Sprawdzenie, czy pola istnieją
    assert "prediction" in data
    assert "probability_survived" in data
    
    # Typy danych
    assert isinstance(data["prediction"], int)
    assert isinstance(data["probability_survived"], float)
    
    # Porównanie predykcji z oczekiwaną
    assert data["prediction"] == expected_prediction

# Testy złych danych (walidacja)
@pytest.mark.parametrize(
    "payload",
    [
        {"Pclass": 5, "Sex": "male", "Age": 22, "Fare": 7.25},      # Pclass poza zakresem
        {"Pclass": 2, "Sex": "unknown", "Age": 22, "Fare": 7.25},   # Sex niepoprawne
        {"Pclass": 2, "Sex": "female", "Age": -1, "Fare": 7.25},    # Age < 0
        {"Pclass": 2, "Sex": "male", "Age": 30, "Fare": -10},       # Fare < 0
    ]
)
def test_predict_invalid(payload):
    response = httpx.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 422  # FastAPI zwraca 422 dla błędnych danych
