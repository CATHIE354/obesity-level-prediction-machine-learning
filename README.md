# Prediction of Obesity Levels Based on Lifestyle and Physical Condition Using Machine Learning

## Project Overview

This project predicts obesity levels based on lifestyle habits, eating habits, and physical condition data using machine learning. The original notebook was restructured into a modular Python project with separate files for data preprocessing, model training, evaluation, and visualization.

## Dataset

The dataset used in this project is:

`ObesityDataSet_raw_and_data_sinthetic.csv`

The target variable is:

`NObeyesdad`

This column represents the obesity level category.

## Project Structure

```text
obesity-level-prediction-machine-learning/
│
├── main.py
├── config.py
├── requirements.txt
├── README.md
│
├── data/
│   └── ObesityDataSet_raw_and_data_sinthetic.csv
│
├── notebooks/
│   └── prediction-of-obesity-levels.ipynb
│
├── preprocessing/
│   └── preprocessing.py
│
├── models/
│   └── model_training.py
│
├── evaluation/
│   └── evaluation.py
│
├── utils/
│   └── visualization.py
│
├── outputs/
│   └── figures/
│
└── tests/
    ├── conftest.py
    ├── test_data_loader.py
    ├── test_model_training.py
    └── test_preprocessing.py