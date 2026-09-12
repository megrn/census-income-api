"""Send one POST request to a deployed API."""

import os

import requests


API_URL = os.getenv(
    "API_URL",
    "https://census-income-api-rikq.onrender.com/predict",
)

payload = {
    "age": 52,
    "workclass": "Self-emp-inc",
    "fnlgt": 209642,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 15000,
    "capital-loss": 0,
    "hours-per-week": 60,
    "native-country": "United-States",
}


if __name__ == "__main__":
    response = requests.post(API_URL, json=payload, timeout=30)
    print(f"status code: {response.status_code}")
    print(response.json())
