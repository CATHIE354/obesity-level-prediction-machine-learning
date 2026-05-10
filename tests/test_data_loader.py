from preprocessing.preprocessing import load_data
from config import DATA_PATH


def test_load_data():
    dataframe = load_data(DATA_PATH)

    assert dataframe is not None
    assert len(dataframe) > 0
    assert "NObeyesdad" in dataframe.columns