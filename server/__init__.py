import os
import threading

import time
import atexit

from flask import Flask, current_app

import mlflow
from mlflow.pyfunc import load_model

from .config import flask_config


def load_configured_model(app: Flask=None):

    app = app or current_app._get_current_object()
    
    config_name = os.getenv('FLASK_ENV', 'development')
    model_name = app.config.get('MLFLOW_MODEL_NAME')
    
    if not model_name:
        app.logger.error('MLFLOW_MODEL_NAME is not configured.')
        app.model = None
        return
    
    model_stage = 'staging' if config_name in {'development', 'testing'} else 'production'
    model_uri = f"models:/{model_name}@{model_stage}"

    try:
        app.model = load_model(model_uri=model_uri)
        app.logger.info(f'{model_name} model loaded successfully in {model_stage}')
    
    except mlflow.exceptions.MlflowException as e:
        app.logger.error(f'Error loading {model_name} model in {model_stage}: {e}')
        app.model = None


def start_model_reload_thread(app: Flask, interval_seconds=300):
    def reload_loop():
        while True:
            try:
                with app.app_context():
                    app.logger.info("Periodic model reload started")
                    load_configured_model(app)
            except Exception as e:
                app.logger.error(f"Error during periodic model reload: {e}")
            time.sleep(interval_seconds)

    thread = threading.Thread(target=reload_loop, daemon=True)
    thread.start()

    def cleanup():
        app.logger.info("Shutting down model reload thread (daemon thread will exit automatically)")

    atexit.register(cleanup)


def create_app(config_name=os.getenv('FLASK_ENV', 'development')):
    app = Flask('Iris Inference Server')

    app.config.from_prefixed_env()
    app.config.from_object(flask_config[config_name])

    load_configured_model(app)

    from .api import api as api_bp
    app.register_blueprint(api_bp)

    start_model_reload_thread(app)

    return app
