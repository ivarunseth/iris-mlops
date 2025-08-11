"""Tests for src.evaluate module."""

from unittest.mock import patch, MagicMock
import pytest

from src.evaluate import evaluate_registered, _evaluate_by_version, evaluate_latest
from src.settings import Settings


@patch("src.evaluate.load_iris_dataset")
@patch("src.evaluate.mlflow.pyfunc.load_model")
def test_evaluate_registered_returns_metrics(mock_load_model, mock_load_data):
    """It should return accuracy and macro-F1 for a given alias."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [0, 1]
    mock_load_model.return_value = mock_model
    mock_load_data.return_value = ([[1, 2], [3, 4]], [0, 1])

    metrics = evaluate_registered(alias="production", model_name="iris_classifier")
    assert metrics["alias"] == "production"
    assert "accuracy" in metrics
    assert "f1_macro" in metrics


@patch("src.evaluate.load_iris_dataset")
@patch("src.evaluate.mlflow.pyfunc.load_model")
def test_evaluate_by_version_returns_metrics(mock_load_model, mock_load_data):
    """It should evaluate a model by version number."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [0, 1]
    mock_load_model.return_value = mock_model
    mock_load_data.return_value = ([[1, 2], [3, 4]], [0, 1])

    metrics = _evaluate_by_version(model_name="iris_classifier", version=5)
    assert metrics["version"] == 5
    assert "accuracy" in metrics
    assert "f1_macro" in metrics


@pytest.mark.skip(reason="Not implemented yet")
@patch("src.evaluate.evaluate_registered")
@patch("src.evaluate.MlflowClient")
def test_evaluate_latest_with_alias(mock_client_cls, _mock_eval_registered):
    """It should evaluate using an alias when provided."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_version = MagicMock()
    mock_version.version = "1"
    mock_version.alias = "production"

    mock_client.get_model_version_by_alias.return_value = mock_version

    settings = Settings()
    result = evaluate_latest(settings=settings, alias="production")

    assert result['version'] == '1'
    assert result['accuracy'] > 0.9


@patch("src.evaluate._evaluate_by_version")
@patch("src.evaluate.MlflowClient")
def test_evaluate_latest_without_alias(mock_client_cls, mock_eval_by_version):
    """It should evaluate the latest version if no alias is given."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client
    mock_version = MagicMock()
    mock_version.version = "9"
    mock_client.search_model_versions.return_value = [mock_version]
    mock_eval_by_version.return_value = {"version": 9, "accuracy": 1.0, "f1_macro": 1.0}

    settings = Settings()
    result = evaluate_latest(settings=settings, alias=None)

    assert result["version"] == 9
    mock_eval_by_version.assert_called_once_with(settings.registered_model_name, 9)


@pytest.mark.skip(reason="Not implemented yet")
def test_evaluate_latest_alias_not_found():
    """It should raise ValueError if alias not found."""
    with patch("src.evaluate.MlflowClient") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.get_registered_model.return_value.aliases = {}
        with pytest.raises(ValueError):
            evaluate_latest(alias="nonexistent")
