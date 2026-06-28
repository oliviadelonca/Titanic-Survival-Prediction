## Overview

This project builds and compares a Random Forest and a Logistic Regression model to predict Titanic passenger survival. It covers the full sklearn workflow: preprocessing pipelines with leakage prevention, stratified cross-validation to handle class imbalance, and hyperparameter tuning via GridSearchCV. 

---

## Dataset

The dataset is loaded directly from seaborn's built-in datasets. The data dictionary is as follows.

| Variable     | Definition                                          |
|:-------------|:----------------------------------------------------|
| `survived`   | Survived? 0 = No, 1 = Yes *(target)*               |
| `pclass`     | Ticket class (1st, 2nd, 3rd)                        |
| `sex`        | Sex                                                 |
| `age`        | Age in years                                        |
| `sibsp`      | Number of siblings / spouses aboard                 |
| `parch`      | Number of parents / children aboard                 |
| `fare`       | Passenger fare                                      |
| `class`      | Ticket class (string form of `pclass`)              |
| `who`        | Man, Woman, or Child                                |
| `adult_male` | Boolean – adult male passenger                      |
| `alone`      | Boolean – travelling alone                          |

---

## Methodology

### 1. Preprocessing

* Numerical features are imputed with the median and scaled with `StandardScaler`
* Categorical features are imputed with the most frequent value and encoded with `OneHotEncoder`
* All transformations are wrapped in a `ColumnTransformer` inside a `sklearn.Pipeline`

### 2. Modelling
* Both models are tuned with `GridSearchCV` using 5-fold stratified cross-validation
* For the Random Forest, the search covers `n_estimators`, `max_depth`, and `min_samples_split`
* For Logistic Regression, it covers regularisation type (`penalty` L1 / L2) and `class_weight`


### 3. Evaluation

* Model performance is assessed via a classification report (precision, recall, F1) and a confusion matrix on the held-out test set
* Feature importances are extracted from the Random Forest, and coefficient magnitudes from the Logistic Regression, to compare what each model relies on

---

## Technologies used

Data manipulation and exploration are handled with **pandas** and **seaborn**, while visualisations are produced with **matplotlib**. All modelling is built on **scikit-learn**, making use of `Pipeline`, `ColumnTransformer`, `GridSearchCV`, `RandomForestClassifier`, and `LogisticRegression`.
