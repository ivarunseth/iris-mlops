from settings import Settings
from src.train import main

def test_train_runs():
    best = main(Settings())
    assert best["algorithm"] in {"logistic_regression", "random_forest"}
    assert best["acc"] > 0.8
