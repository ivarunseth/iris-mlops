import os

from flask import Blueprint, current_app, request, Response, g, abort


api = Blueprint('api', __name__, url_prefix='/api')


from .utils import *


@api.before_request
def before_request():
    g.t = timestamp()


@api.get('/health')
def health():
    response = {'model': current_app.config['MLFLOW_MODEL_NAME'], 'environment': \
            os.getenv('FLASK_ENV', 'development'), 'status': 'ok'}

    if not getattr(current_app, 'model', None):
        response['status'] = 'not ok'

    return response, 200


@api.post('/predict')
def predict():
    if not getattr(current_app, 'model', None):
        abort(503, f"{current_app.config['MLFLOW_MODEL_NAME']} model is not available for inference.")

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
    return get_metrics(), 200


@api.errorhandler(400)
def bad_request(error):
    return {'error': error.description}, 400


@api.errorhandler(503)
def service_unavailable(error):
    return {'error': error.description}, 503
    

@api.errorhandler(Exception)
def exception(error):
    return format_exception(error), 500


@api.after_request
def after_request(response: Response):
    latency = (timestamp() - g.t) * 1000

    append_request(request.path, response.status_code, latency)
    
    environ = {k: v for k, v in request.environ.items() if isinstance(v, (str, bytes))}

    if 'wsgi.input' in request.environ and request.is_json:
        environ['wsgi.input'] = request.get_json(force=True, silent=True)

    current_app.logger.info({
        'headers': {k: v for k, v in request.headers.items()},
        'request': environ,
        'response': response.get_json(),
        'status': response.status,
        'latency': latency
    })

    return response