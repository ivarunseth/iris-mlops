import pytest
from unittest.mock import patch
from src import stage


@patch("src.stage.mlflow.client.MlflowClient.transition_model_version_stage")
def test_stage_model(mock_transition):
    mock_transition.return_value = None
    stage.stage_model("model_name", "1", "Production")
    mock_transition.assert_called_once()
