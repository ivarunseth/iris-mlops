"""Tests for src.register module."""

from unittest.mock import patch, MagicMock

import pytest

from src.register import register_best_model, get_best_run_id
from src.settings import Settings


@patch("src.register.mlflow.register_model")
def test_register_best_model_calls_mlflow_register(mock_register_model):
    """It should call mlflow.register_model with the correct URI and return version."""
    mock_register_model.return_value.version = "5"
    settings = Settings()
    best = {"run_id": "123abc"}
    version = register_best_model(best, settings)
    assert version == "5"
    mock_register_model.assert_called_once()
    _, kwargs = mock_register_model.call_args
    assert settings.registered_model_name in kwargs["name"]


@patch("src.register.MlflowClient")
def test_get_best_run_id_returns_expected_id(mock_client_cls):
    """It should return the run_id of the best run."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "42"
    mock_client.get_experiment_by_name.return_value = mock_experiment

    mock_run = MagicMock()
    mock_run.info.run_id = "best123"
    mock_run.data.metrics.get.return_value = 0.99
    mock_client.search_runs.return_value = [mock_run]

    run_id = get_best_run_id(Settings())
    assert run_id == "best123"


@patch("src.register.MlflowClient")
def test_get_best_run_id_raises_when_no_runs(mock_client_cls):
    """It should raise ValueError if no runs are found."""
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "42"
    mock_client.get_experiment_by_name.return_value = mock_experiment

    mock_client.search_runs.return_value = []

    settings = Settings()
    settings.experiment_name = "empty_experiment"

    with pytest.raises(ValueError):
        get_best_run_id(settings)
