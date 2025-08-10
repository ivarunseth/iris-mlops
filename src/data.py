"""
Data loading utilities for the Iris dataset.

Provides a helper function to load the Iris dataset from scikit-learn
and return it as a pandas DataFrame and Series.
"""

from typing import Tuple

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.utils import Bunch


def load_iris_dataset(as_frame: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load the Iris dataset.

    Args:
        as_frame (bool): Whether to return the data as a pandas DataFrame/Series.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing:
            - features_df: DataFrame with feature columns
            - target_series: Series with target values
    """
    iris: Bunch = load_iris(as_frame=as_frame)  # type: ignore
    features_df = iris.data  # pylint: disable=no-member
    target_series = iris.target  # pylint: disable=no-member
    features_df.columns = iris.feature_names  # pylint: disable=no-member
    return features_df, target_series
