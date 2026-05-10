from preprocessing.preprocessing import load_data, preprocess_data
from config import DATA_PATH, TARGET_COLUMN


def test_preprocess_data():
    dataframe = load_data(DATA_PATH)

    x_train, x_test, y_train, y_test = preprocess_data(
        dataframe,
        TARGET_COLUMN
    )

    assert len(x_train) > 0
    assert len(x_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0
    assert x_train.shape[1] == x_test.shape[1]