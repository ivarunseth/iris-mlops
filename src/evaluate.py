"""
Evaluation script for MLflow-registered Iris classification models.

This module provides functions to evaluate the performance of models
registered in MLflow, either by alias (e.g., 'production', 'staging'),
or by retrieving the latest registered model version if no alias is provided.
"""

from typing import Dict, Any
import argparse

from sklearn.metrics import accuracy_score, f1_score

import mlflow
from mlflow.tracking import MlflowClient

from .settings import Settings
from .data import load_iris_dataset


def evaluate_registered(alias: str, model_name: str) -> Dict[str, Any]:
    """
    Evaluate a registered MLflow model by alias.

    Args:
        alias (str): Alias of the model (e.g., 'production', 'staging').
        model_name (str): Name of the registered MLflow model.

    Returns:
        Dict[str, Any]: A dictionary containing alias, accuracy, and macro F1 score.
    """
    uri = f"models:/{model_name}@{alias}"
    model = mlflow.pyfunc.load_model(uri)

    features_df, target_series = load_iris_dataset()
    preds = model.predict(features_df)

    return {
        "alias": alias,
        "accuracy": accuracy_score(target_series, preds),
        "f1_macro": f1_score(target_series, preds, average="macro"),
    }


def _evaluate_by_version(model_name: str, version: int) -> Dict[str, Any]:
    """
    Helper function to evaluate a model by version number.
    """
    uri = f"models:/{model_name}/{version}"
    model = mlflow.pyfunc.load_model(uri)

    features_df, target_series = load_iris_dataset()
    preds = model.predict(features_df)

    return {
        "version": version,
        "accuracy": accuracy_score(target_series, preds),
        "f1_macro": f1_score(target_series, preds, average="macro"),
    }


def evaluate_latest(settings: Settings = Settings(), alias: str = "production") -> Dict[str, Any]:
    """
    Evaluate the latest model version assigned to a given alias in MLflow.

    If alias is None or empty, evaluate the latest registered version instead.

    Args:
        settings (Settings): Configuration settings including MLflow tracking URI.
        alias (str): Model alias to evaluate (default: 'production').

    Returns:
        Dict[str, Any]: Evaluation metrics for the specified model alias or latest version.
    """
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    client = MlflowClient()
    model_name = settings.registered_model_name

    if alias:
        # Get all versions assigned to this stage (alias)
        try:
            model = client.get_model_version_by_alias(model_name, alias)
        except mlflow.exceptions.RestException:
            print(f"No versions found for model '{model_name}' at alias/stage '{alias}'")
            return 

        # Pick the latest version by version number
        metrics = _evaluate_by_version(model_name, model.version)
        print(f"Evaluation for alias '{alias}' (v{model.version}): {metrics}")
        return metrics

    # No alias → evaluate latest registered version (any stage)
    # The stages param will be deprecated soon,
    # but no alternative yet; for now we can get all versions
    all_versions = client.search_model_versions(f"name='{model_name}'")
    if not all_versions:
        raise ValueError(f"No versions found for model '{model_name}'.")

    latest_version = max(all_versions, key=lambda v: int(v.version))
    metrics = _evaluate_by_version(model_name, int(latest_version.version))
    print(f"Evaluation for latest model version (v{latest_version.version}): {metrics}")
    return metrics


def main() -> None:
    """Parse CLI arguments and evaluate a model alias or the latest registered version."""
    parser = argparse.ArgumentParser(description="Evaluate a registered MLflow model.")
    parser.add_argument(
        "alias",
        nargs="?",
        default=None,
        help="Model alias to evaluate (default: None → latest registered version)",
    )
    args = parser.parse_args()
    evaluate_latest(alias=args.alias)


if __name__ == "__main__":
    main()
