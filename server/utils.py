import time
import traceback

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
