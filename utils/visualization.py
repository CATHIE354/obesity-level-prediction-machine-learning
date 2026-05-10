import os
import matplotlib.pyplot as plt
import seaborn as sns
from config import OUTPUT_FIGURES_PATH, TARGET_COLUMN


def create_output_folder():
    """
    Create output folder for saved figures.
    """
    os.makedirs(OUTPUT_FIGURES_PATH, exist_ok=True)


def plot_target_distribution(dataframe):
    """
    Plot and save the distribution of obesity levels.
    """
    create_output_folder()

    plt.figure(figsize=(10, 6))
    sns.countplot(data=dataframe, x=TARGET_COLUMN)
    plt.title("Distribution of Obesity Levels")
    plt.xlabel("Obesity Level")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_FIGURES_PATH}/obesity_level_distribution.png")
    plt.close()


def plot_confusion_matrix(matrix):
    """
    Plot and save the confusion matrix.
    """
    create_output_folder()

    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_FIGURES_PATH}/confusion_matrix.png")
    plt.close()


def plot_feature_importance(model, feature_names):
    """
    Plot and save feature importance from the Random Forest model.
    """
    create_output_folder()

    importance = model.feature_importances_

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importance, y=feature_names)
    plt.title("Feature Importance")
    plt.xlabel("Importance Score")
    plt.ylabel("Features")
    plt.tight_layout()

    plt.savefig(f"{OUTPUT_FIGURES_PATH}/feature_importance.png")
    plt.close()