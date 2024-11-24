# PRODIGY_DS_03
# Decision Tree Classifier for Predicting Customer Purchase Behavior

## Overview
This project involves building a **Decision Tree Classifier** to predict whether a customer will purchase a product or service based on their demographic and behavioral data. The dataset used is the **Bank Marketing Dataset** from the UCI Machine Learning Repository. The solution incorporates data cleaning, exploratory data analysis (EDA), feature engineering, and predictive modeling.

---

## Table of Contents
1. [Project Description](#project-description)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)

---

## Project Description
The goal of this project is to create a machine learning model that can classify whether a customer will subscribe to a term deposit based on various features such as age, job, marital status, and balance. The primary steps include:
- Data Cleaning and Preparation
- Exploratory Data Analysis (EDA)
- Training a Decision Tree Classifier
- Evaluating the model's performance
- Visualizing the Decision Tree

---

## Installation
To run this project locally, you'll need Python and the following libraries installed:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `imbalanced-learn`

You can install the dependencies using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn
```
## Dataset

The dataset used is the **Bank Marketing Dataset**, which contains **45,211 rows** and **17 columns**. You can download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing).

### Features
1. **Demographic Data**: `age`, `job`, `marital`, `education`
2. **Behavioral Data**: `balance`, `housing`, `loan`, `contact`
3. **Target Variable**: `y` (whether the customer subscribed to a term deposit)

---

## Methodology

### 1. Data Cleaning
- Checked for missing values and duplicates (none found).
- Encoded categorical variables using **Label Encoding** and **One-Hot Encoding**.

### 2. Exploratory Data Analysis (EDA)
- Visualized distributions of numerical and categorical variables.
- Analyzed relationships between features and the target variable.
- Computed a correlation matrix to understand feature relationships.

### 3. Modeling
- Implemented a **Decision Tree Classifier** using `scikit-learn`.
- Handled class imbalance using **SMOTE** (Synthetic Minority Oversampling Technique).
- Tuned hyperparameters using **GridSearchCV** for optimal performance.

### 4. Evaluation
- Assessed the model using metrics such as **Accuracy**, **Confusion Matrix**, and **Classification Report**.
- Plotted and interpreted the decision tree for visualization purposes.

---

## Results

1. **Key Feature**: The `duration` variable (length of the last contact) was the most significant predictor.
2. **Accuracy**: Achieved a high accuracy score using a tuned Decision Tree Classifier.
3. The confusion matrix and classification report indicated strong predictive performance.

---
## Contributing

Contributions are welcome! If you'd like to improve this project:

1. Fork the repository.
2. Create a branch for your feature or bug fix.
3. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the [LICENSE.md](LICENSE.md) file for details.
