from typing import Dict, Any

import argparse

import mlflow
from mlflow.tracking import MlflowClient

from sklearn.metrics import accuracy_score, f1_score

from settings import Settings
from src.data import load_iris_dataset


def evaluate_registered(alias: str, model_name: str) -> Dict[str, Any]:
    """Evaluate a model by alias (e.g., 'staging', 'production')."""
    uri = f"models:/{model_name}@{alias}"
    model = mlflow.pyfunc.load_model(uri)

    X, y = load_iris_dataset(as_frame=True)
    preds = model.predict(X)

    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, average="macro")

    return {"alias": alias, "accuracy": acc, "f1_macro": f1}


def evaluate_latest(settings: Settings = Settings(), alias: str = "production"):
    """Evaluate the latest model registered under the given alias."""
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = MlflowClient()
    model_name = settings.registered_model_name

    # Check if alias exists
    registered_model = client.get_registered_model(model_name)
    aliases = registered_model.aliases or {}

    if alias not in aliases:
        raise ValueError(f"No alias '{alias}' found for model '{model_name}'.")

    version = aliases[alias]
    metrics = evaluate_registered(alias=alias, model_name=model_name)

    print(f"Evaluation for alias '{alias}' (v{version}): {metrics}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "alias",
        default="production",
        help="Model alias to evaluate (default: production)",
    )
    args = parser.parse_args()
    evaluate_latest(alias=args.alias)


if __name__ == "__main__":
    main()
