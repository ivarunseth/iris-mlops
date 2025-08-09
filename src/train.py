import mlflow
import mlflow.sklearn

from datetime import datetime
from typing import Dict, Any

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from settings import Settings
from src.data import load_iris_dataset


def main(settings: Settings) -> Dict[str, Any]:
    if settings.mlflow_tracking_uri:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_name)

    X, y = load_iris_dataset(as_frame=True)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=settings.test_size, random_state=settings.random_state, stratify=y
    )

    # Candidate estimators
    candidates = {
        "logistic_regression": LogisticRegression(max_iter=settings.max_iter, random_state=settings.random_state),
        "random_forest": RandomForestClassifier(random_state=settings.random_state),
    }

    preprocess = Pipeline([("scaler", StandardScaler())])
    best = {"name": None, "acc": -1.0, "f1": -1.0, "run_id": None}

    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", estimator)])

        # Retrieve parameter grid from settings
        param_grid = settings.param_grids.get(name, {})

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            cv=3,
            scoring="accuracy",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)

        # Timestamp for run name
        run_name = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with mlflow.start_run(run_name=run_name) as run:
            best_pipe = grid.best_estimator_

            # Log algorithm name & best parameters
            mlflow.log_param("algorithm", name)
            mlflow.log_params(grid.best_params_)

            # Metrics
            y_pred = best_pipe.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            f1_macro = f1_score(y_val, y_pred, average="macro")
            mlflow.log_metrics({"val_accuracy": acc, "val_f1_macro": f1_macro})

            # Signature + example
            input_example = X_train.iloc[:2, :]
            signature = mlflow.models.infer_signature(X_train, best_pipe.predict(X_train))

            mlflow.sklearn.log_model(
                best_pipe,
                name="model",
                input_example=input_example,
                signature=signature,
            )

            print({
                "event": "trained",
                "model": name,
                "best_params": grid.best_params_,
                "val_accuracy": acc,
                "val_f1_macro": f1_macro
            })

            if acc > best["acc"]:
                best.update({
                    "name": name,
                    "acc": acc,
                    "f1": f1_macro,
                    "run_id": run.info.run_id
                })

    print({"event": "best_model", "best": best})


if __name__ == "__main__":
    main(settings=Settings())
