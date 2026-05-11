from config import DATA_PATH, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE

from preprocessing.preprocessing import (
    load_data,
    preprocess_data,
    inspect_dataset,
    clean_data,
    add_bmi_feature
)

from models.model_training import train_random_forest
from evaluation.evaluation import evaluate_model
from utils.visualization import (
    plot_target_distribution,
    plot_confusion_matrix,
    plot_feature_importance
)


def main():
    """
    Main workflow for obesity level prediction.
    """
    print("Loading dataset...")
    dataframe = load_data(DATA_PATH)
    
    print("Inspecting dataset...")
    inspect_dataset(dataframe, TARGET_COLUMN)

    print("Cleaning dataset...")
    dataframe = clean_data(dataframe)

    print("Adding BMI feature...")
    dataframe = add_bmi_feature(dataframe)

    print("Creating target distribution graph...")
    plot_target_distribution(dataframe)

    print("Preprocessing data...")
    x_train, x_test, y_train, y_test = preprocess_data(
        dataframe,
        TARGET_COLUMN,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("Training Random Forest model...")
    model = train_random_forest(x_train, y_train, random_state=RANDOM_STATE)

    print("Evaluating model...")
    accuracy, report, matrix, predictions = evaluate_model(model, x_test, y_test)

    print("\nModel Accuracy:")
    print(accuracy)

    print("\nClassification Report:")
    print(report)

    print("Saving confusion matrix and feature importance graphs...")
    plot_confusion_matrix(matrix)
    plot_feature_importance(model, x_train.columns)

    print("\nProject completed successfully.")


if __name__ == "__main__":
    main()
