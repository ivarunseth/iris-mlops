"""Entry point for running the Flask application."""
from server import create_app


application = app = create_app()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
