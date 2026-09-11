"""FastAPI app for census income prediction."""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from starter.ml.data import process_data
from starter.train_model import CAT_FEATURES, train_and_save_model


ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
LB_PATH = MODEL_DIR / "lb.pkl"

app = FastAPI(
    title="Census Income Classifier",
    description="Predict whether a person's income is above or below 50K.",
    version="1.0.0",
)


class CensusRecord(BaseModel):
    """Input schema for a single census record."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
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
        },
    )

    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")


def load_pickle(path):
    """Load a pickle artifact."""
    with open(path, "rb") as file:
        return pickle.load(file)


def load_artifacts():
    """Load model artifacts, training them first if they are absent."""
    if not (MODEL_PATH.exists() and ENCODER_PATH.exists() and LB_PATH.exists()):
        train_and_save_model()
    return load_pickle(MODEL_PATH), load_pickle(ENCODER_PATH), load_pickle(LB_PATH)


@app.get("/")
async def root() -> dict[str, str]:
    """Return a welcome message."""
    return {"message": "Welcome to the Census Income Classifier API."}


@app.post("/predict")
async def predict(record: CensusRecord) -> dict[str, str | int]:
    """Predict whether income is >50K or <=50K for a census record."""
    model, encoder, lb = load_artifacts()
    input_df = pd.DataFrame([record.model_dump(by_alias=True)])
    X, _, _, _ = process_data(
        input_df,
        categorical_features=CAT_FEATURES,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    prediction = int(model.predict(X)[0])
    salary = lb.inverse_transform(np.array([prediction]))[0]
    return {"prediction": prediction, "salary": salary}
