from dotenv import load_dotenv
load_dotenv()

import os


class Config(object):
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'top-secret-key')

    MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    MLFLOW_MODEL_NAME = os.getenv('MLFLOW_MODEL_NAME', 'iris_classifier')

    REQUEST_STATS_WINDOW = 15


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    pass


class TestingConfig(Config):
    TESTING = True


flask_config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig
}