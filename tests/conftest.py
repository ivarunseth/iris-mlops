"""Common pytest fixtures for tests."""
from unittest.mock import patch, MagicMock  # standard library first
import pytest  # third party

from sklearn.model_selection import train_test_split

from src.settings import Settings
from src.data import load_iris_dataset
from src.train import DatasetSplit


@pytest.fixture
def small_settings_fixture():
    """Fixture providing small parameter grid for faster tests."""
    settings = Settings()
    settings.param_grids = {
        "logistic_regression": {"model__C": [0.1]},
        "decision_tree": {"model__max_depth": [2]},
    }
    return settings


@pytest.fixture
def dataset_split_fixture(settings):
    """Fixture creating a small dataset split."""
    features, target = load_iris_dataset(as_frame=True)
    x_train, x_val, y_train, y_val = train_test_split(
        features,
        target,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=target
    )
    return DatasetSplit(x_train, x_val, y_train, y_val)


@pytest.fixture
def mock_mlflow_fixture():
    """Patch all MLflow methods used in train.py so no real tracking occurs."""
    with patch("src.train.mlflow") as mock:
        mock.start_run.return_value.__enter__.return_value.info.run_id = "fake_run_id"
        mock.start_run.return_value.__exit__ = MagicMock()
        mock.set_tracking_uri = MagicMock()
        mock.set_experiment = MagicMock()
        mock.log_param = MagicMock()
        mock.log_params = MagicMock()
        mock.log_metrics = MagicMock()
        mock.log_artifact = MagicMock()
        mock.sklearn.log_model = MagicMock()
        mock.models.infer_signature = MagicMock(return_value="fake_signature")
        yield mock
