import os
import matplotlib.pyplot as plt
from config import OUTPUT_FIGURES_PATH, TARGET_COLUMN


def create_output_folder():
    """
    Create output folder for figures.
    """
    os.makedirs(OUTPUT_FIGURES_PATH, exist_ok=True)


def plot_target_distribution(dataframe):
    """
    Plot obesity level distribution.
    """
    create_output_folder()

    counts = dataframe[TARGET_COLUMN].value_counts()

    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts.values)
    plt.title("Distribution of Obesity Levels")
    plt.xlabel("Obesity Level")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FIGURES_PATH}/obesity_level_distribution.png")
    plt.close()


def plot_confusion_matrix(matrix):
    """
    Plot confusion matrix.
    """
    create_output_folder()

    plt.figure(figsize=(8, 6))
    plt.imshow(matrix)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.colorbar()

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, matrix[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FIGURES_PATH}/confusion_matrix.png")
    plt.close()


def plot_feature_importance(model, feature_names):
    """
    Plot feature importance.
    """
    create_output_folder()

    importance = model.feature_importances_

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance)
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FIGURES_PATH}/feature_importance.png")
    plt.close()