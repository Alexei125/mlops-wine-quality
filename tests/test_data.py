import pandas as pd
import os

# Путь относительно корня репозитория
DATA_PATH = os.getenv("WINE_DATA_PATH", "data/winequality-red.csv")


def test_data_exists():
    assert os.path.exists(DATA_PATH), f"Data not found at {DATA_PATH}"


def test_data_shape():
    df = pd.read_csv(DATA_PATH)
    assert df.shape[0] > 0
    assert "quality" in df.columns
    assert df.shape[1] == 12
