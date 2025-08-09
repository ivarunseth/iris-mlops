import os
import sys

from flask import Flask

import mlflow
from mlflow.pyfunc import load_model

from .config import flask_config


def create_app(config_name=os.getenv('FLASK_ENV', 'development')):
    app = Flask('Iris Inference Server')

    app.config.from_prefixed_env()
    app.config.from_object(flask_config[config_name])

    try:
        mlflow.set_tracking_uri(app.config['MLFLOW_TRACKING_URI'])

        model_name = app.config['MLFLOW_MODEL_NAME']
        model_stage = 'staging' if config_name in {'development', 'testing'} else 'production'
        model_uri = f'models:/{model_name}@{model_stage}'
        app.model = load_model(model_uri=model_uri)
    
        app.logger.info(f'{model_name} model loaded successfully in {model_stage}')

    except mlflow.exceptions.MlflowException as e:
        app.logger.error(f'Error loading {model_name} model in {model_stage}: {e}')
        sys.exit(1)

    from .api import api as api_bp
    app.register_blueprint(api_bp)

    return app
