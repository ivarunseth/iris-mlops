from src.data import load_iris_dataframe

def test_load_iris_dataframe():
    X, y = load_iris_dataframe()
    assert X.shape[1] == 4  # 4 numeric features[1][2][5][8][11]
    assert len(X) == 150 and len(y) == 150  # 150 samples[2][5][8]
