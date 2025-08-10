"""
API blueprint for the Iris Inference Server.

Provides endpoints to:
- Check service health
- Make predictions using the loaded MLflow model
- Retrieve service metrics

Includes request lifecycle hooks for logging and error handling.
"""

import os

from flask import Blueprint, current_app, request, Response, g, abort

from .utils import timestamp, validate, classes, features, \
    append_request, get_metrics, format_exception


api = Blueprint('api', __name__, url_prefix='/api')


@api.before_request
def before_request():
    """
    Record the start timestamp of the incoming request.

    Stores the timestamp in Flask's `g` context to measure request latency later.
    """
    g.t = timestamp()


@api.get('/health')
def health():
    """
    Health check endpoint.

    Returns the current model name, environment, and service status.
    Status is 'ok' if the model is loaded, otherwise 'not ok'.

    Returns:
        tuple: JSON response dict and HTTP status code 200.
    """
    response = {
        'model': current_app.config['MLFLOW_MODEL_NAME'],
        'environment': os.getenv('FLASK_ENV', 'development'),
        'status': 'ok'
    }

    if not getattr(current_app, 'model', None):
        response['status'] = 'not ok'

    return response, 200


@api.post('/predict')
def predict():
    """
    Prediction endpoint.

    Accepts JSON input with features, validates the request,
    and returns model predictions along with class labels and feature names.

    Returns:
        tuple: JSON response with predictions, classes, and features; HTTP status 200.

    Raises:
        503 ServiceUnavailable: If the model is not loaded.
        400 BadRequest: If the input JSON is invalid.
    """
    if not getattr(current_app, 'model', None):
        abort(503, f"{current_app.config['MLFLOW_MODEL_NAME']} \
              model is not available for inference.")

    data = request.get_json(cache=True, force=True, silent=True)

    if not validate(data):
        abort(400, 'Invalid request JSON data.')

    predictions = current_app.model.predict(data['inputs'])
    response = {
        'predictions': [int(p) for p in predictions],
        'classes': [classes[int(p)] for p in predictions],
        'features': features
    }

    return response, 200


@api.get('/metrics')
def metrics():
    """
    Metrics endpoint.

    Returns monitoring metrics collected for the inference server.

    Returns:
        tuple: JSON metrics data and HTTP status code 200.
    """
    return get_metrics(), 200


@api.errorhandler(400)
def bad_request(error):
    """
    Handle 400 Bad Request errors.

    Args:
        error (HTTPException): The raised HTTP exception.

    Returns:
        tuple: JSON error message and HTTP status code 400.
    """
    return {'error': error.description}, 400


@api.errorhandler(503)
def service_unavailable(error):
    """
    Handle 503 Service Unavailable errors.

    Args:
        error (HTTPException): The raised HTTP exception.

    Returns:
        tuple: JSON error message and HTTP status code 503.
    """
    return {'error': error.description}, 503


@api.errorhandler(Exception)
def exception(error):
    """
    Handle unexpected exceptions.

    Args:
        error (Exception): The raised exception.

    Returns:
        tuple: JSON formatted error details and HTTP status code 500.
    """
    return format_exception(error), 500


@api.after_request
def after_request(response: Response):
    """
    After request handler to log request and response details.

    Measures latency, appends request metrics, and logs request headers,
    environment info, response content, status, and latency.

    Args:
        response (Response): The Flask response object.

    Returns:
        Response: The unchanged response object.
    """
    latency = timestamp() - g.t

    append_request(request.path, response.status_code, latency)
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
