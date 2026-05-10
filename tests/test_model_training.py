from preprocessing.preprocessing import load_data, preprocess_data
from models.model_training import train_random_forest
from config import DATA_PATH, TARGET_COLUMN


def test_train_random_forest():
    dataframe = load_data(DATA_PATH)

    x_train, x_test, y_train, y_test = preprocess_data(
        dataframe,
        TARGET_COLUMN
    )

    model = train_random_forest(x_train, y_train)

    assert model is not None
    assert hasattr(model, "predict")

    predictions = model.predict(x_test)
    assert len(predictions) == len(y_test)