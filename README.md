# Prediction of Obesity Levels Based on Lifestyle and Physical Condition Using Machine Learning

## Project Overview

This project predicts obesity levels based on lifestyle habits, eating habits, and physical condition data using machine learning. The original notebook was restructured into a modular Python project with separate files for data preprocessing, model training, evaluation, visualization, and testing.

## Dataset

The dataset used in this project is:

`ObesityDataSet_raw_and_data_sinthetic.csv`

The target variable is:

`NObeyesdad`

This column represents the obesity level category.

## Project Structure

```text
obesity-level-prediction-machine-learning/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── data/
│   └── ObesityDataSet_raw_and_data_sinthetic.csv
├── preprocessing/
│   └── preprocessing.py
├── models/
│   └── model_training.py
├── evaluation/
│   └── evaluation.py
├── utils/
│   └── visualization.py
├── outputs/
│   └── figures/
└── tests/
    ├── conftest.py
    ├── test_data_loader.py
    ├── test_model_training.py
    └── test_preprocessing.py
```

## Installation

Install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run

Run the main program:

```bash
python main.py
```

## How to Run Tests

Run the unit tests:

```bash
pytest
```

## Machine Learning Model

This project uses a Random Forest Classifier to predict obesity levels.

## Evaluation

The model is evaluated using:

- Accuracy score
- Classification report
- Confusion matrix

The model achieved an accuracy of approximately 95.74% on the test dataset.

## Output

The generated figures are saved in:

```text
outputs/figures/
```

The generated figures include:

- Obesity level distribution graph
- Confusion matrix
- Feature importance graph

## Main Modules

### `main.py`

Runs the complete machine learning workflow, including loading the dataset, preprocessing, model training, evaluation, and visualization.

### `config.py`

Stores fixed project settings such as dataset path, target column name, test size, random state, and output folder path.

### `preprocessing/preprocessing.py`

Handles data loading, feature-target splitting, categorical encoding, and train-test splitting.

### `models/model_training.py`

Contains the Random Forest model training function.

### `evaluation/evaluation.py`

Evaluates the trained model using accuracy score, classification report, and confusion matrix.

### `utils/visualization.py`

Generates and saves visualizations such as obesity level distribution, confusion matrix, and feature importance graph.

### `tests/`

Contains unit tests to check data loading, preprocessing, and model training.

## Author

Cathie Valentina Jimmy