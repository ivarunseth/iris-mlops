import pytest
from server import create_app


app = create_app(config_name='testing')


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json.get("status") == "ok"


def test_predict_endpoint(client, monkeypatch):
    class DummyModel:
        def predict(self, X):
            return [42]

    monkeypatch.setattr("src.server.model", DummyModel())
    resp = client.post("/predict", json={"data": [[5.1, 3.5, 1.4, 0.2]]})
    assert resp.status_code == 200
    assert "predictions" in resp.json
