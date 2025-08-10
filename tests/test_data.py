"""Unit tests for the load_iris_dataset function in src.data."""

import pandas as pd
from src.data import load_iris_dataset


def test_load_iris_dataset_as_frame():
    """Test that the function returns a DataFrame and Series when as_frame=True."""
    features, target = load_iris_dataset(as_frame=True)
    assert isinstance(features, pd.DataFrame)
    assert isinstance(target, pd.Series)
    assert features.shape[0] == target.shape[0]


def test_load_iris_dataset_as_numpy():
    """Test that the function returns array-like objects when as_frame=False."""
    features, target = load_iris_dataset(as_frame=False)
    assert hasattr(features, "shape")
    assert hasattr(target, "shape")
    assert features.shape[0] == target.shape[0]
