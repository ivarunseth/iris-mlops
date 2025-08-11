"""Unit tests for training module."""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.train import TrainContext, train_and_log_model, select_best_model, main


def test_train_and_log_model_returns_metrics(dataset_split_fixture, _mock_mlflow_fixture):
    """Test that training and logging a model returns metrics and run_id."""
    ctx = TrainContext(
        algorithm="logistic_regression",
        model=LogisticRegression(max_iter=10),
        preprocess=Pipeline([("scaler", StandardScaler())]),
        param_grid={"model__C": [0.1]},
        train_data=(dataset_split_fixture.x_train, dataset_split_fixture.y_train),
        val_data=(dataset_split_fixture.x_val, dataset_split_fixture.y_val)
    )
    metrics, run_id = train_and_log_model(ctx)
    assert "val_accuracy" in metrics
    assert run_id == "fake_run_id"


def test_select_best_model(dataset_split_fixture, small_settings_fixture, _mock_mlflow_fixture):
    """Test that the best model is selected from candidates."""
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=10),
        "decision_tree": DecisionTreeClassifier()
    }
    preprocess = Pipeline([("scaler", StandardScaler())])
    best_model = select_best_model(candidates, preprocess, \
        dataset_split_fixture, small_settings_fixture)
    assert (
        best_model["algorithm"] in candidates
    ), "Selected algorithm not in candidates list"


def test_main_runs(small_settings_fixture, _mock_mlflow_fixture):
    """Test that main training routine runs successfully."""
    result = main(small_settings_fixture)
    assert "algorithm" in result
    assert result["score"] > 0
