from typing import Tuple

import pandas as pd

from sklearn.datasets import load_iris


def load_iris_dataset(as_frame: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    iris = load_iris(as_frame=as_frame)
    X = iris.data
    y = iris.target
    X.columns = iris.feature_names
    return X, y
