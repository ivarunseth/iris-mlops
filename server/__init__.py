"""
This module sets up the Iris Inference Server Flask application with MLflow model integration.

It provides functions to:
- Load the MLflow model configured via environment variables and Flask app settings.
- Start a background thread to periodically reload the model to ensure it stays up-to-date.
- Create and configure the Flask app with appropriate configurations, 
  API blueprint, and model management.

The module supports multiple environments (development, testing, production) 
and automatically switches model stages accordingly.
"""

import os

from flask import Flask, current_app

import mlflow
from mlflow.pyfunc import load_model

from .api import api as api_bp
from .config import flask_config


def load_configured_model(app: Flask = None):
    """
    Load the MLflow model configured for the Flask application.

    The function tries to load the model based on environment variables and app config.
    It selects the model stage based on the current environment 
    ('development' or 'testing' uses 'staging', otherwise 'production').

    If the model name is not configured or if loading fails, 
    the app.model attribute will be set to None.

    Args:
        app (Flask, optional): The Flask app instance to load the model into.
                               Defaults to the current Flask application context.
    """
    app = app or current_app
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


def create_app(config_name=os.getenv('FLASK_ENV', 'development')):
    """
    Factory function to create and configure the Flask application.

    Loads environment and object-based configurations, loads the MLflow model,
    registers API blueprints, and starts the model reload background thread.

    Args:
        config_name (str, optional): The Flask configuration name to use.

    Returns:
        Flask: The fully configured Flask application instance.
    """
    app = Flask('Iris Inference Server')

    app.config.from_prefixed_env()
    app.config.from_object(flask_config[config_name])

    load_configured_model(app)

    app.register_blueprint(api_bp)

    app.logger.info('Application started')

    return app
