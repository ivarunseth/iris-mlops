from typing import Any, Dict

import mlflow
from mlflow.tracking import MlflowClient

from settings import Settings


def register_best_model(best: Dict[str, Any], settings: Settings) -> str:
    assert best["run_id"], "No run_id found for best model"
    model_uri = f"runs:/{best['run_id']}/model"
    result = mlflow.register_model(model_uri=model_uri, name=settings.registered_model_name)
    return result.version


def get_best_run_id(settings: Settings) -> str:
    """
    Query MLflow for the best run based on val_accuracy.
    """
    if cfg.mlflow_tracking_uri:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=[client.get_experiment_by_name(settings.experiment_name).experiment_id],
        filter_string="",
        order_by=["metrics.val_accuracy DESC"],
        max_results=1
    )

    if not runs:
        raise ValueError(f"No runs found for experiment '{settings.experiment_name}'")

    best_run = runs[0]
    print({
        "event": "best_run_found",
        "run_id": best_run.info.run_id,
        "val_accuracy": best_run.data.metrics.get("val_accuracy")
    })
    return best_run.info.run_id


if __name__ == "__main__":
    cfg = Settings()
    best_run_id = get_best_run_id(cfg)
    version = register_best_model({"run_id": best_run_id}, cfg)
    print({
        "registered_model": cfg.registered_model_name,
        "version": version
    })
