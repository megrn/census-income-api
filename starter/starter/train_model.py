"""Train the census income model and save model artifacts."""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference, train_model


CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "census.csv"
MODEL_DIR = ROOT_DIR / "model"
MODEL_PATH = MODEL_DIR / "model.pkl"
ENCODER_PATH = MODEL_DIR / "encoder.pkl"
LB_PATH = MODEL_DIR / "lb.pkl"
SLICE_OUTPUT_PATH = ROOT_DIR / "slice_output.txt"


def load_data(path=DATA_PATH):
    """Load the census data and strip whitespace from column names and strings."""
    data = pd.read_csv(path)
    data.columns = data.columns.str.strip()

    string_columns = data.select_dtypes(include=["object"]).columns
    data[string_columns] = data[string_columns].apply(lambda col: col.str.strip())
    return data


def save_pickle(obj, path):
    """Persist a Python object with pickle."""
    with open(path, "wb") as file:
        pickle.dump(obj, file)


def load_pickle(path):
    """Load a pickled Python object."""
    with open(path, "rb") as file:
        return pickle.load(file)


def evaluate_slices(model, data, encoder, lb, slice_features=CAT_FEATURES):
    """Calculate model performance for each value of selected categorical features."""
    lines = []
    for feature in slice_features:
        for value in sorted(data[feature].dropna().unique()):
            slice_data = data[data[feature] == value]
            X_slice, y_slice, _, _ = process_data(
                slice_data,
                categorical_features=CAT_FEATURES,
                label="salary",
                training=False,
                encoder=encoder,
                lb=lb,
            )
            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)
            lines.append(
                f"{feature}: {value} | "
                f"precision: {precision:.4f} | "
                f"recall: {recall:.4f} | "
                f"fbeta: {fbeta:.4f}"
            )
    return "\n".join(lines)


def train_and_save_model():
    """Train the model, evaluate it, and save all required artifacts."""
    data = load_data()
    train, test = train_test_split(
        data, test_size=0.20, random_state=42, stratify=data["salary"]
    )

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    MODEL_DIR.mkdir(exist_ok=True)
    save_pickle(model, MODEL_PATH)
    save_pickle(encoder, ENCODER_PATH)
    save_pickle(lb, LB_PATH)

    slice_output = evaluate_slices(model, test, encoder, lb)
    SLICE_OUTPUT_PATH.write_text(slice_output, encoding="utf-8")

    print(f"precision: {precision:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"fbeta: {fbeta:.4f}")
    print(f"wrote slice metrics to {SLICE_OUTPUT_PATH}")
    return precision, recall, fbeta


if __name__ == "__main__":
    train_and_save_model()
