"""Train multiple classifiers on the Iris dataset with MLflow + DVC tracking."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple
from datetime import datetime
from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import mlflow
import mlflow.sklearn

from .settings import Settings
from .data import load_iris_dataset


@dataclass
class DatasetSplit:
    """Container for train/validation features and targets."""
    x_train: Any
    x_val: Any
    y_train: Any
    y_val: Any


@dataclass
class TrainContext:
    """Container for all training data, parameters, and preprocessing for a single model run."""
    algorithm: str
    model: Any
    preprocess: Pipeline
    param_grid: dict
    train_data: Tuple[Any, Any]
    val_data: Tuple[Any, Any]


def plot_confusion_matrix(cm: np.ndarray, class_names: list, out_path: Path):
    """Plot and save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def train_and_log_model(ctx: TrainContext) -> Tuple[Dict[str, float], str, Any]:
    """Train a single model, log to MLflow, and return metrics + run_id + best_model."""
    x_train, y_train = ctx.train_data
    x_val, y_val = ctx.val_data

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        pipeline = Pipeline([("preprocess", ctx.preprocess), ("model", ctx.model)])
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=ctx.param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid.fit(x_train, y_train)

        run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with mlflow.start_run(run_name=run_name) as run:
            best_pipeline = grid.best_estimator_
            mlflow.log_param("algorithm", ctx.algorithm)
            mlflow.log_params(grid.best_params_)

            predictions = best_pipeline.predict(x_val)
            metrics = {
                "val_accuracy": accuracy_score(y_val, predictions),
                "val_precision_macro": precision_score(y_val, predictions, average="macro"),
                "val_precision_weighted": precision_score(y_val, predictions, average="weighted"),
                "val_recall_macro": recall_score(y_val, predictions, average="macro"),
                "val_recall_weighted": recall_score(y_val, predictions, average="weighted"),
                "val_f1_macro": f1_score(y_val, predictions, average="macro"),
                "val_f1_weighted": f1_score(y_val, predictions, average="weighted"),
            }
            mlflow.log_metrics(metrics)

            # Save classification report as TXT
            report_txt_path = tmp_path / "classification_report.txt"
            with open(report_txt_path, "w", encoding="utf-8") as f:
                f.write(classification_report(y_val, predictions))
            mlflow.log_artifact(str(report_txt_path))

            # Save confusion matrix as PNG
            cm = confusion_matrix(y_val, predictions)
            cm_png_path = tmp_path / "confusion_matrix.png"
            plot_confusion_matrix(cm, class_names=list(np.unique(y_val)), out_path=cm_png_path)
            mlflow.log_artifact(str(cm_png_path))

            input_example = x_train.iloc[:2, :]
            signature = mlflow.models.infer_signature(x_train, best_pipeline.predict(x_train))
            mlflow.sklearn.log_model(best_pipeline, name="model", input_example=input_example, signature=signature)

        return metrics, run.info.run_id


def select_best_model(candidates, preprocess, dataset: DatasetSplit, settings):
    """Train each candidate model, evaluate, and return the best model metadata."""
    best_model_meta = {
        "algorithm": None,
        "score": -1.0,
        "acc": -1.0,
        "f1": -1.0,
        "precision": -1.0,
        "recall": -1.0,
        "run_id": None,
    }

    for algorithm, model in candidates.items():
        ctx = TrainContext(
            algorithm=algorithm,
            model=model,
            preprocess=preprocess,
            param_grid=settings.param_grids.get(algorithm, {}),
            train_data=(dataset.x_train, dataset.y_train),
            val_data=(dataset.x_val, dataset.y_val),
        )
        metrics, run_id = train_and_log_model(ctx)

        composite_score = (
            metrics["val_accuracy"]
            + metrics["val_f1_macro"]
            + metrics["val_precision_macro"]
            + metrics["val_recall_macro"]
        ) / 4

        if composite_score > best_model_meta["score"]:
            best_model_meta.update({
                "algorithm": algorithm,
                "score": composite_score,
                "acc": metrics["val_accuracy"],
                "f1": metrics["val_f1_macro"],
                "precision": metrics["val_precision_macro"],
                "recall": metrics["val_recall_macro"],
                "run_id": run_id,
            })

    return best_model_meta


def main(settings: Settings) -> Dict[str, Any]:
    """Train candidate models, log metrics and artifacts to MLflow, return best model metadata."""
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    features, target = load_iris_dataset()
    dataset = DatasetSplit(
        *train_test_split(
            features,
            target,
            test_size=settings.test_size,
            random_state=settings.random_state,
            stratify=target,
        )
    )

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=settings.max_iter,
            random_state=settings.random_state,
        ),
        "random_forest": RandomForestClassifier(random_state=settings.random_state),
        "decision_tree": DecisionTreeClassifier(random_state=settings.random_state),
    }

    preprocess = Pipeline([("scaler", StandardScaler())])

    best_model_meta = select_best_model(candidates, preprocess, dataset, settings)

    print({"event": "best_model", "best": best_model_meta})
    return best_model_meta


if __name__ == "__main__":
    main(settings=Settings())
