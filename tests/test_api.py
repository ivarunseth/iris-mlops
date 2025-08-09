import json

from server import create_app


app = create_app(config_name='testing')


def test_health_endpoint():
    with app.test_client() as c:
        rv = c.get("/api/health")
        assert rv.status_code == 200

def test_predict_endpoint():
    # Load a local model by mocking mlflow if needed;
    # here we rely on a running registry in CI or skip if not available.
    payload = {"inputs": [[5.1, 3.5, 1.4, 0.2]]}
    with app.test_client() as c:
        rv = c.post("/api/predict", data=json.dumps(payload), content_type="application/json")
        assert rv.status_code == 200
        data = rv.get_json()
        assert "predictions" in data and "classes" in data
