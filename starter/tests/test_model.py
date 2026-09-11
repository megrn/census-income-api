import numpy as np

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference, train_model
from starter.train_model import CAT_FEATURES, load_data


def test_process_data_returns_encoded_arrays():
    data = load_data().head(20)
    X, y, encoder, lb = process_data(
        data, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == y.shape[0] == 20
    assert encoder is not None
    assert lb is not None


def test_train_model_and_inference():
    data = load_data().head(100)
    X, y, _, _ = process_data(
        data, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    model = train_model(X, y)
    preds = inference(model, X[:5])

    assert len(preds) == 5
    assert set(preds).issubset({0, 1})


def test_compute_model_metrics_known_values():
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 1.0
    assert recall == 0.5
    assert round(fbeta, 2) == 0.67


def test_load_data_strips_whitespace():
    data = load_data()

    assert "salary" in data.columns
    assert not any(column.startswith(" ") or column.endswith(" ") for column in data.columns)
    object_values = data.select_dtypes(include=["object"]).stack()
    assert not object_values.str.startswith(" ").any()
    assert not object_values.str.endswith(" ").any()
