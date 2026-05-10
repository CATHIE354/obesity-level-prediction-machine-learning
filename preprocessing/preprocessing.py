import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(file_path):
    """
    Load dataset from a CSV file.
    """
    return pd.read_csv(file_path)


def split_features_target(dataframe, target_column):
    """
    Split dataset into features and target.
    """
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]

    return features, target


def encode_categorical_columns(dataframe):
    """
    Encode categorical columns using LabelEncoder.
    """
    encoded_dataframe = dataframe.copy()
    encoders = {}

    for column in encoded_dataframe.select_dtypes(include=["object", "category"]).columns:
        encoder = LabelEncoder()
        encoded_dataframe[column] = encoder.fit_transform(encoded_dataframe[column])
        encoders[column] = encoder

    return encoded_dataframe, encoders


def preprocess_data(dataframe, target_column, test_size=0.2, random_state=42):
    """
    Prepare train and test data.
    """
    features, target = split_features_target(dataframe, target_column)
    encoded_features, encoders = encode_categorical_columns(features)

    x_train, x_test, y_train, y_test = train_test_split(
        encoded_features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target
    )

    return x_train, x_test, y_train, y_test