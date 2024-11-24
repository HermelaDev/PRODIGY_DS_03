# Decision Tree Classifier for Bank Deposit Prediction
## Overview

This project aims to predict whether a customer will subscribe to a deposit product using a dataset of customer attributes such as age, job, marital status, and account balance. A Decision Tree classifier is used to make predictions based on the given features.

## Table of Contents

1. [Installation](#installation)
2. [Data Collection](#data-collection)
3. [Data Cleaning](#data-cleaning)
4. [Model Building](#model-building)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)
8. [Contact](#contact)

## Installation

To run this project, ensure you have the following Python libraries installed:
---bash
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import warnings
from kaggle.api.kaggle_api_extended import KaggleApi
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

## Data Collection

The dataset used in this project is the **Bank Marketing Dataset** from Kaggle, available at [rouseguy/bankbalanced](https://www.kaggle.com/datasets/rouseguy/bankbalanced). It includes customer information along with whether they subscribed to a deposit (`yes` or `no`).

## Data Cleaning

The data cleaning process involved several steps to prepare the dataset for analysis and modeling:

- **Handling Missing Data**: Missing values were handled by replacing them with the most frequent value in the respective column.
- **Encoding Categorical Variables**: Categorical features like `job` and `education` were encoded into numerical values using `LabelEncoder`.
- **Feature Scaling**: Features such as `age` and `balance` were scaled using `StandardScaler` to ensure the model treats them equally.
- **Data Splitting**: The dataset was split into training and testing sets using `train_test_split`.

## Model Building

A **Decision Tree Classifier** was used to predict whether a customer will subscribe to a deposit. The model was trained on the cleaned and preprocessed dataset.

- **Training**: The model was trained on the training set with hyperparameters tuned for better performance.
- **Evaluation**: The model's performance was evaluated using accuracy, precision, recall, and F1-score.


## Results

- **Accuracy**: 81%
- **Precision**: 0.80
- **Recall**: 0.82
- **F1-Score**: 0.81


## Feature Importance

The most important features in predicting whether a customer subscribes to a deposit were:

- Duration of last contact
- Balance
- Previous contacts

<p align="center">
  <img src="images/model_performance.jpg" alt="Model Performance" width="300"/>
</p>

## Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository.
2. Create a branch for your feature or bug fix.
3. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.

## Contact

If you have any questions or need further assistance, feel free to reach out:
Hermela Seltanu
LinkedIn: [Hermela Seltanu](https://www.linkedin.com/in/hermelaseltanu/)
