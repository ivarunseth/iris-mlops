"""
API blueprint for the Iris Inference Server.

Provides endpoints to:
- Check service health
- Make predictions using the loaded MLflow model
- Retrieve service metrics

Includes request lifecycle hooks for logging and error handling.
"""

import os
import time
from flask import Blueprint, current_app, request, Response, g, abort

from .utils import (
    timestamp, validate, classes, features,
    format_exception, increment_prediction_count,
    mark_model_loaded, get_metrics,
    REQUEST_COUNT, REQUEST_LATENCY, ERROR_COUNT
)

api = Blueprint('api', __name__, url_prefix='/api')


@api.before_request
def before_request():
    """Store the start timestamp for request latency calculation."""
    g.start_time = time.time()


@api.get('/health')
def health():
    """Health check endpoint."""
    model_loaded = getattr(current_app, 'model', None) is not None
    mark_model_loaded(current_app.config['MLFLOW_MODEL_NAME'], model_loaded)

    response = {
        'model': current_app.config['MLFLOW_MODEL_NAME'],
        'environment': os.getenv('FLASK_ENV', 'development'),
        'status': 'ok' if model_loaded else 'not ok'
    }
    return response, 200


@api.post('/predict')
def predict():
    """Prediction endpoint."""
    if not getattr(current_app, 'model', None):
        return format_exception(ValueError(f"{current_app.config['MLFLOW_MODEL_NAME']} \
              model is not available for inference.")), 503

    data = request.get_json(cache=True, force=True, silent=True)

    if not validate(data):
        return format_exception(ValueError('Invalid request JSON data.')), 400

    predictions = current_app.model.predict(data['inputs'])

    increment_prediction_count(current_app.config['MLFLOW_MODEL_NAME'])

    response = {
        'predictions': [int(p) for p in predictions],
        'classes': [classes[int(p)] for p in predictions],
        'features': features
    }

    return response, 200


@api.get('/metrics')
def metrics():
    """Prometheus metrics endpoint."""
    data, content_type = get_metrics()
    return Response(data, status=200, content_type=content_type)


@api.errorhandler(400)
def bad_request(error):
    ERROR_COUNT.labels(exception_type="BadRequest", endpoint=request.path).inc()
    return {'error': error.description}, 400


@api.errorhandler(503)
def service_unavailable(error):
    ERROR_COUNT.labels(exception_type="ServiceUnavailable", endpoint=request.path).inc()
    return {'error': error.description}, 503


@api.errorhandler(Exception)
def internal_server_error(error):
    ERROR_COUNT.labels(exception_type="InternalServerError", endpoint=request.path).inc()
    return format_exception(error), 500


@api.after_request
def after_request(response: Response):
    """Record Prometheus metrics after each request."""
    latency = time.time() - getattr(g, "start_time", time.time())
    endpoint = request.endpoint or "unknown"
    method = request.method
    status_code = response.status_code

    REQUEST_COUNT.labels(method=method, endpoint=endpoint, http_status=status_code).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)

    environ = {k: v for k, v in request.environ.items() if isinstance(v, (str, bytes))}
    if 'wsgi.input' in request.environ and request.is_json:
        environ['wsgi.input'] = request.get_json(force=True, silent=True)

    current_app.logger.info({
        'headers': dict(request.headers.items()),
        'request': environ,
        'response': response.get_json(),
        'status': response.status,
        'latency': latency
    })

    return response
