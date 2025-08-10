"""
Configuration settings for the Iris classification ML pipeline.

This module defines the `Settings` dataclass, which holds experiment,
model, and MLflow configuration parameters, as well as hyperparameter grids
for training.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class Settings:
    """
    Global configuration for the Iris classification project.

    Attributes:
        experiment_name (str): Name of the MLflow experiment.
        registered_model_name (str): Name of the MLflow registered model.
        test_size (float): Fraction of the dataset used for testing.
        random_state (int): Random seed for reproducibility.
        training_params (Dict[str, Any]): Model training parameters such as
            maximum iterations and number of estimators.
        mlflow_tracking_uri (Optional[str]): URI for the MLflow tracking server.
        param_grids (Dict[str, Dict[str, Any]]): Hyperparameter grids for model tuning.
    """
    experiment_name: str = "iris-classification"
    registered_model_name: str = "iris_classifier"
    test_size: float = 0.2
    random_state: int = 42
    training_params: Dict[str, Any] = field(default_factory=lambda: {
        "max_iter": 200,
        "n_estimators": 200
    })
    mlflow_tracking_uri: Optional[str] = os.getenv(
        "MLFLOW_TRACKING_URI",
        "http://localhost:5000"
    )

    # Hyperparameter grids for GridSearchCV
    param_grids: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1, 10],
            "model__solver": ["lbfgs"],
        },
        "random_forest": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10],
        },
        "decision_tree": {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__criterion": ["gini", "entropy"],
        },
    })
