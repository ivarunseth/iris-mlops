"""
Tests for MLflow model staging and promotion utilities in src.stage.
"""

from unittest.mock import patch, MagicMock
import pytest
from src import stage


@patch("time.sleep", return_value=None)
def test_wait_until_ready_success(_mock_sleep):
    """Test wait_until_ready returns when model version becomes READY."""

    mock_client = MagicMock()
    statuses = ["PENDING", "READY"]
    mock_client.get_model_version.side_effect = [
        MagicMock(status=statuses[0]),
        MagicMock(status=statuses[1]),
    ]

    stage.wait_until_ready(mock_client, "model_name", "1", timeout_s=5)
    assert mock_client.get_model_version.call_count == 2


@patch("time.sleep", return_value=None)
def test_wait_until_ready_timeout(_mock_sleep):
    """Test wait_until_ready raises TimeoutError if status not READY in time."""

    mock_client = MagicMock()
    mock_client.get_model_version.return_value = MagicMock(status="PENDING")

    with pytest.raises(TimeoutError):
        stage.wait_until_ready(mock_client, "model_name", "1", timeout_s=1)


@patch("src.stage.mlflow.tracking.MlflowClient")
@patch("src.stage.wait_until_ready")
def test_set_alias_default_version(mock_wait_ready, mock_mlflow_client):
    """Test set_alias assigns alias to latest version if version not provided."""

    mock_client = MagicMock()
    mock_mlflow_client.return_value = mock_client

    # Mock search_model_versions to return two versions
    mock_client.search_model_versions.return_value = [
        MagicMock(version="1"),
        MagicMock(version="2"),
    ]

    version = stage.set_alias("staging")

    # It should pick the highest version "2"
    assert version == "2"
    mock_wait_ready.assert_called_once_with(mock_client, \
        stage.Settings().registered_model_name, "2")
    mock_client.set_registered_model_alias.assert_called_once_with(
        name=stage.Settings().registered_model_name,
        alias="staging",
        version="2"
    )


@patch("src.stage.mlflow.tracking.MlflowClient")
@patch("src.stage.wait_until_ready")
def test_set_alias_with_version(mock_wait_ready, mock_mlflow_client):
    """Test set_alias assigns alias to specified version."""

    mock_client = MagicMock()
    mock_mlflow_client.return_value = mock_client

    version = stage.set_alias("production", version="5")

    assert version == "5"
    mock_wait_ready.assert_called_once_with(mock_client, \
        stage.Settings().registered_model_name, "5")
    mock_client.set_registered_model_alias.assert_called_once_with(
        name=stage.Settings().registered_model_name,
        alias="production",
        version="5"
    )


@patch("src.stage.mlflow.tracking.MlflowClient")
def test_delete_alias_calls_client(mock_mlflow_client):
    """Test delete_alias calls MLflow client to delete alias."""

    mock_client = MagicMock()
    mock_mlflow_client.return_value = mock_client

    stage.delete_alias("staging")

    mock_client.delete_registered_model_alias.assert_called_once_with(
        stage.Settings().registered_model_name,
        "staging"
    )


@pytest.mark.skip(reason="Not implemented yet")
@patch("src.stage.set_alias")
@patch("src.stage.mlflow.tracking.MlflowClient")
def test_promote_latest_auto_no_alias(mock_set_alias, mock_mlflow_client):
    """Test promote_latest_auto promotes model with no alias to staging."""

    mock_client = MagicMock()
    mock_mlflow_client.return_value = mock_client

    # Setup versions and model aliases
    version_obj = MagicMock(version="3")
    mock_client.search_model_versions.return_value = [version_obj]
    mock_client.get_registered_model.return_value.aliases = {}

    mock_set_alias.return_value = "3"

    stage.promote_latest_auto()

    mock_set_alias.assert_called_once_with("staging", version="3")


@pytest.mark.skip(reason="Not implemented yet")
@patch("src.stage.set_alias")
@patch("src.stage.mlflow.tracking.MlflowClient")
def test_promote_latest_auto_staging_to_production(mock_set_alias, mock_mlflow_client):
    """Test promote_latest_auto promotes staging alias to production."""

    mock_client = MagicMock()
    mock_mlflow_client.return_value = mock_client

    version_obj = MagicMock(version="7")
    mock_client.search_model_versions.return_value = [version_obj]
    mock_client.get_registered_model.return_value.aliases = {"staging": "7"}

    mock_set_alias.return_value = "7"

    stage.promote_latest_auto()

    mock_set_alias.assert_called_once_with("production", version="7")


@patch("src.stage.mlflow.tracking.MlflowClient")
def test_promote_latest_auto_already_production(mock_mlflow_client):
    """Test promote_latest_auto raises error if already in production."""

    mock_client = MagicMock()
    mock_mlflow_client.return_value = mock_client

    version_obj = MagicMock(version="9")
    mock_client.search_model_versions.return_value = [version_obj]
    # aliases include production already on this version
    mock_client.get_registered_model.return_value.aliases = {"production": "9"}

    with pytest.raises(ValueError, match="Already in production"):
        stage.promote_latest_auto()


@pytest.mark.skip(reason="Not implemented yet")
@patch("src.stage.delete_alias")
@patch("src.stage.mlflow.tracking.MlflowClient")
def test_demote_latest_auto_remove_production(mock_delete_alias, mock_mlflow_client):
    """Test demote_latest_auto removes production alias."""

    mock_client = MagicMock()
    mock_mlflow_client.return_value = mock_client

    # aliases with production on version 5
    mock_client.get_registered_model.return_value.aliases = {"production": "5", "staging": "4"}

    stage.demote_latest_auto()

    mock_delete_alias.assert_called_once_with("production")


@pytest.mark.skip(reason="Not implemented yet")
@patch("src.stage.delete_alias")
@patch("src.stage.mlflow.tracking.MlflowClient")
def test_demote_latest_auto_remove_staging(mock_delete_alias, mock_mlflow_client):
    """Test demote_latest_auto removes staging alias if no production."""

    mock_client = MagicMock()
    mock_mlflow_client.return_value = mock_client

    # aliases with only staging
    mock_client.get_registered_model.return_value.aliases = {"staging": "2"}

    stage.demote_latest_auto()

    mock_delete_alias.assert_called_once_with("staging")


@patch("src.stage.mlflow.tracking.MlflowClient")
def test_demote_latest_auto_no_aliases_raises(mock_mlflow_client):
    """Test demote_latest_auto raises error if no aliases found."""

    mock_client = MagicMock()
    mock_mlflow_client.return_value = mock_client

    mock_client.get_registered_model.return_value.aliases = {}

    with pytest.raises(ValueError, match="No aliases found"):
        stage.demote_latest_auto()
