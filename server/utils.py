"""
Utility functions and data structures for the Iris Inference Server.

Includes:
- Feature and class metadata
- Request validation
- Metrics tracking and aggregation
- Exception formatting
"""

from dataclasses import dataclass
from typing import List

import time
import traceback
import threading

from collections import defaultdict, deque

from flask import current_app


features = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

classes = ["setosa", "versicolor", "virginica"]

# Module-private globals for metrics tracking
_metrics_lock = threading.Lock()
_TOTAL_REQUESTS = 0
_TOTAL_LATENCY = 0.0
_status_counts = defaultdict(int)
_endpoint_counts = defaultdict(int)
_request_timestamps = deque()


@dataclass(frozen=True)
class PredictRequest:
    """
    Data structure representing a prediction request.

    Attributes:
        inputs (List[List[float]]): List of feature vectors for prediction.
    """
    inputs: List[List[float]]


def validate(payload) -> bool:
    """
    Validate the input payload for prediction.

    Args:
        payload (dict): The JSON payload from the client.

    Returns:
        bool: True if payload is valid, False otherwise.

    Valid payloads must contain 'inputs' key with a non-empty list of feature vectors.
    Each feature vector can be either a list of numbers matching features length
    or a dict with keys matching the feature names.
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
    """
    Get the current time as a floating point number of seconds since the epoch.

    Returns:
        float: Current timestamp.
    """
    return time.time()


def format_exception(e: Exception):
    """
    Format an exception into a JSON-serializable dictionary.

    Includes the error message, exception type, and optionally the traceback
    if the Flask app is in debug or testing mode.

    Args:
        e (Exception): The exception to format.

    Returns:
        dict: A dictionary with error details.
    """
    exception = {'error': str(e), 'type': type(e).__name__}

    if current_app.debug or current_app.testing:
        exception.update({'traceback': traceback.format_exc()})

    return exception


def append_request(endpoint: str, status_code: int, latency: float):
    """
    Append a completed request's metadata for metrics tracking.

    This updates global counters and timestamps while ensuring thread safety.

    Args:
        endpoint (str): The API endpoint accessed.
        status_code (int): HTTP status code returned.
        latency (float): Time taken to handle the request in seconds.
    """
    global _TOTAL_REQUESTS, _TOTAL_LATENCY  # pylint: disable=global-statement

    now = time.time()
    with _metrics_lock:
        _TOTAL_REQUESTS += 1
        _TOTAL_LATENCY += latency
        _status_counts[status_code] += 1
        _endpoint_counts[endpoint] += 1
        _request_timestamps.append(now)

        cutoff = now - current_app.config['REQUEST_STATS_WINDOW']
        while _request_timestamps and _request_timestamps[0] < cutoff:
            _request_timestamps.popleft()


def get_metrics():
    """
    Compute and return aggregated metrics about recent requests.

    Returns:
        dict: Metrics including total requests, average latency (ms),
              requests per second, status code counts, and endpoint counts.
    """
    with _metrics_lock:
        average_latency = (_TOTAL_LATENCY / _TOTAL_REQUESTS) if _TOTAL_REQUESTS > 0 else 0
        requests_per_second = len(_request_timestamps) / current_app.config['REQUEST_STATS_WINDOW']

        return {
            "total_requests": _TOTAL_REQUESTS,
            "average_latency": average_latency * 1000,  # Convert to milliseconds
            "requests_per_second": requests_per_second,
            "status_code_counts": dict(_status_counts),
            "endpoint_counts": dict(_endpoint_counts)
        }
