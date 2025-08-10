import pytest
from unittest.mock import patch
from src import evaluate


@patch("src.evaluate.mlflow.pyfunc.load_model")
def test_evaluate_model(mock_load_model):
    class DummyModel:
        def predict(self, X):
            return [0, 1]

    mock_load_model.return_value = DummyModel()
    preds = evaluate.evaluate_model("fake_model_uri", [[1, 2], [3, 4]])
    assert preds == [0, 1]
