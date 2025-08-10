import pytest
from unittest.mock import patch
from src import register


@patch("src.register.mlflow.register_model")
def test_register_model_function(mock_register):
    mock_register.return_value = None
    register.register_model("runs:/fake_run/model", "test_model")
    mock_register.assert_called_once()
