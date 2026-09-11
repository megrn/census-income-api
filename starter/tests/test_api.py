from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


LOW_INCOME_EXAMPLE = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}


HIGH_INCOME_EXAMPLE = {
    "age": 52,
    "workclass": "Self-emp-not-inc",
    "fnlgt": 209642,
    "education": "HS-grad",
    "education-num": 9,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States",
}


def test_get_root():
    response = client.get("/")

    assert response.status_code == 200
    assert "Welcome" in response.json()["message"]


def test_post_predict_low_income():
    response = client.post("/predict", json=LOW_INCOME_EXAMPLE)

    assert response.status_code == 200
    assert response.json()["salary"] == "<=50K"


def test_post_predict_high_income():
    response = client.post("/predict", json=HIGH_INCOME_EXAMPLE)

    assert response.status_code == 200
    assert response.json()["salary"] == ">50K"
