import os

from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass(frozen=True)
class Settings:
    experiment_name: str = "iris-classification"
    registered_model_name: str = "iris_classifier"
    test_size: float = 0.2
    random_state: int = 42
    max_iter: int = 200
    n_estimators: int = 200
    mlflow_tracking_uri: Optional[str] = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')

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
    })
