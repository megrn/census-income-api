"""Send one POST request to a deployed API."""

import os

import requests


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

payload = {
    "age": 52,
    "workclass": "Self-emp-inc",
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


if __name__ == "__main__":
    response = requests.post(API_URL, json=payload, timeout=30)
    print(f"status code: {response.status_code}")
    print(response.json())
