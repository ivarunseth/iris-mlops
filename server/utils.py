"""
Utility functions and data structures for the Iris Inference Server.

Includes:
- Feature and class metadata
- Request validation
- Prometheus metrics tracking
- Exception formatting
"""

from dataclasses import dataclass
from typing import List
import time
import traceback

from flask import current_app, request
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

# Model metadata
features = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]
classes = ["setosa", "versicolor", "virginica"]

# --- Prometheus metrics ---
REQUEST_COUNT = Counter(
    "iris_api_requests_total",
    "Total HTTP requests processed by the inference API",
    ["method", "endpoint", "http_status"]
)

REQUEST_LATENCY = Histogram(
    "iris_api_request_latency_seconds",
    "Request latency in seconds",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10)
)

PREDICTION_COUNT = Counter(
    "iris_api_predictions_total",
    "Total number of predictions made",
    ["model"]
)

ERROR_COUNT = Counter(
    "iris_api_exceptions_total",
    "Total exceptions encountered",
    ["exception_type", "endpoint"]
)

MODEL_LOADED = Gauge(
    "iris_api_model_loaded",
    "1 if model loaded and ready, 0 otherwise",
    ["model"]
)


@dataclass(frozen=True)
class PredictRequest:
    """Data structure representing a prediction request."""
    inputs: List[List[float]]


def validate(payload) -> bool:
    """
    Validate the input payload for prediction.
    """
    if not isinstance(payload, dict) or "inputs" not in payload:
        return False

    inputs = payload["inputs"]
    if not isinstance(inputs, list) or not inputs:
        return False

    first_row = inputs[0]

    if isinstance(first_row, list):
        return all(
            isinstance(row, list)
            and len(row) == len(features)
            and all(isinstance(x, (int, float)) for x in row)
            for row in inputs
        )

    if isinstance(first_row, dict):
        return all(
            isinstance(row, dict)
            and set(row.keys()) == set(features)
            and all(isinstance(row[k], (int, float)) for k in features)
            for row in inputs
        )

    return False


def timestamp():
    """Return the current time in seconds."""
    return time.time()


def format_exception(e: Exception):
    """
    Format an exception into a JSON-serializable dictionary.
    Also increments Prometheus ERROR_COUNT metric.
    """
    ERROR_COUNT.labels(exception_type=type(e).__name__, endpoint=current_app.view_functions.get(request.endpoint, "unknown")).inc()
    exception = {'error': str(e), 'type': type(e).__name__}
    if current_app.debug or current_app.testing:
        exception.update({'traceback': traceback.format_exc()})
    return exception


def append_request(method: str, endpoint: str, status_code: int, latency: float):
    """
    Record request metrics for Prometheus.
    """
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=status_code).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)


def mark_model_loaded(model_name: str, loaded: bool):
    """
    Mark a model as loaded or unloaded for Prometheus.
    """
    MODEL_LOADED.labels(model=model_name).set(1 if loaded else 0)


def increment_prediction_count(model_name: str):
    """
    Increment prediction counter for a given model.
    """
    PREDICTION_COUNT.labels(model=model_name).inc()


def get_metrics():
    """
    Generate Prometheus metrics exposition format output.

    Returns:
        tuple: (response_body: bytes, content_type: str)
    """
    return generate_latest(), CONTENT_TYPE_LATEST
