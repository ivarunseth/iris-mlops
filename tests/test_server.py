"""Tests for the server API endpoints."""

import pytest
from server import create_app

# pylint: disable=too-few-public-methods
class DummyModel:
    """A model that returns constant predictions."""

    def predict(self, _x):
        """Return a fixed prediction list."""
        return [0]


@pytest.fixture(name="test_client")
def fixture_client():
    """Fixture to provide a Flask test client and app instance."""
    app = create_app(config_name="testing")
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c, app


def test_predict_no_model(test_client):
    """Test that /api/predict returns 503 if no model is loaded."""
    c, _ = test_client
    resp = c.post("/api/predict", json={"inputs": [[1, 2, 3, 4]]})
    assert resp.status_code == 503
    assert "not available" in resp.get_json()["error"]


def test_predict_invalid_json(test_client):
    """Test that /api/predict returns 400 if input JSON is invalid."""
    c, app = test_client
    app.model = DummyModel()
    resp = c.post("/api/predict", json={"bad_key": "oops"})
    assert resp.status_code == 400
    assert "Invalid request" in resp.get_json()["error"]


def test_predict_exception(test_client):
    """Test that /api/predict returns 500 on model prediction errors."""
    # pylint: disable=too-few-public-methods
    class BadModel:
        """A model that raises an error when predicting."""

        def predict(self, _x):
            """Raise a RuntimeError."""
            raise RuntimeError("boom")

    c, app = test_client
    app.model = BadModel()
    resp = c.post("/api/predict", json={"inputs": [[1, 2, 3, 4]]})
    assert resp.status_code == 500
    assert "error" in resp.get_json()


def test_metrics_endpoint(test_client):
    """Test that /api/metrics returns a dictionary."""
    c, _ = test_client
    resp = c.get("/api/metrics")
    assert resp.status_code == 200
    assert isinstance(resp.get_json(), dict)
