"""
Data loading utilities for the Iris dataset with DVC tracking.

Loads the Iris dataset from a local CSV (tracked by DVC) and returns
it as a pandas DataFrame (features) and Series (target).
"""

from typing import Tuple
import pandas as pd


def load_iris_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Iris dataset from a local CSV file tracked with DVC.

    Args:
        as_frame (bool): Whether to return the data as a pandas DataFrame/Series.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (features_df, target_series)
    """
    df = pd.read_csv('data/iris.csv')
    features_df = df.drop(columns=["target"])
    target_series = df["target"]
    return features_df, target_series
