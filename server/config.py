"""
Configuration module for the Iris Inference Server.

Defines configuration classes for different environments:
- Development
- Production
- Testing

Loads environment variables from a `.env` file if available.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# pylint: disable=too-few-public-methods
class Config:
    """
    Base configuration class with default settings.

    Attributes:
        DEBUG (bool): Flag to enable/disable debug mode.
        TESTING (bool): Flag to enable/disable testing mode.
        SECRET_KEY (str): Secret key for session management.
        MLFLOW_TRACKING_URI (str): URI for MLflow tracking server.
        MLFLOW_MODEL_NAME (str): Name of the MLflow model to use.
        REQUEST_STATS_WINDOW (int): Time window for request statistics in seconds.
    """
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'top-secret-key')

    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    MLFLOW_MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME', 'iris_classifier')

    REQUEST_STATS_WINDOW = int(os.getenv('REQUEST_STATS_WINDOW', '15'))


# pylint: disable=too-few-public-methods
class DevelopmentConfig(Config):
    """
    Development environment configuration.

    Enables debug mode.
    """
    DEBUG = True


# pylint: disable=too-few-public-methods
class ProductionConfig(Config):
    """
    Production environment configuration.

    Inherits settings from base Config without modifications.
    """


# pylint: disable=too-few-public-methods
class TestingConfig(Config):
    """
    Testing environment configuration.

    Enables testing mode.
    """
    TESTING = True


flask_config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}
