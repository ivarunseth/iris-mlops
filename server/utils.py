import time
import traceback
import threading

from collections import defaultdict, deque

from flask import current_app

from dataclasses import dataclass
from typing import List


features = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)"
]

classes = ["setosa", "versicolor", "virginica"]

_metrics_lock = threading.Lock()
_total_requests = 0
_total_latency = 0.0
_status_counts = defaultdict(int)
_endpoint_counts = defaultdict(int)
_request_timestamps = deque()


@dataclass(frozen=True)
class PredictRequest:
    inputs: List[List[float]]


def validate(payload) -> bool:
    if not isinstance(payload, dict) or "inputs" not in payload:
        return False

    inputs = payload["inputs"]
    if not isinstance(inputs, list) or len(inputs) == 0:
        return False

    first_row = inputs[0]

    if isinstance(first_row, list):
        for row in inputs:
            if not isinstance(row, list) or len(row) != len(features):
                return False
            if not all(isinstance(x, (int, float)) for x in row):
                return False
        return True

    elif isinstance(first_row, dict):
        for row in inputs:
            if not isinstance(row, dict):
                return False
            if set(row.keys()) != set(features):
                return False
            if not all(isinstance(row[k], (int, float)) for k in features):
                return False
        return True

    return False


def timestamp():
    return time.time()


def format_exception(e: Exception):
    exception = {'error': str(e), 'type': type(e).__name__}
    
    if current_app.debug or current_app.testing:
        exception.update({'traceback': traceback.format_exc()})

    return exception


def append_request(endpoint: str, status_code: int, latency: float):
    global _total_requests, _total_latency

    now = time.time()
    with _metrics_lock:
        _total_requests += 1
        _total_latency += latency
        _status_counts[status_code] += 1
        _endpoint_counts[endpoint] += 1
        _request_timestamps.append(now)

        cutoff = now - current_app.config['REQUEST_STATS_WINDOW']
        while _request_timestamps and _request_timestamps[0] < cutoff:
            _request_timestamps.popleft()


def get_metrics():
    with _metrics_lock:
        average_latency = (_total_latency / _total_requests) if _total_requests > 0 else 0
        requests_per_second = len(_request_timestamps) / current_app.config['REQUEST_STATS_WINDOW']

        return {
            "total_requests": _total_requests,
            "average_latency": average_latency * 1000,
            "requests_per_second": requests_per_second,
            "status_code_counts": dict(_status_counts),
            "endpoint_counts": dict(_endpoint_counts)
        }
