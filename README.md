# Heart Disease Prediction — XGBoost + SHAP Explainability

**Domain:** Healthcare | Explainable AI  
**Author:** Aqib Hussain

## Problem
Heart disease is the leading cause of death globally. Early
prediction from clinical features enables preventive intervention.
This model predicts heart disease presence and explains each
decision using SHAP values.

## Dataset
UCI Heart Disease (Cleveland) — 303 patients, 14 clinical features.
Download: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

## Results
| Metric    | Score |
|-----------|-------|
| Accuracy  | 92.1% |
| AUC-ROC   | 0.96  |
| F1-Score  | 0.92  |
| Precision | 0.93  |
| Recall    | 0.91  |
| CV (5-fold)| 91.8% ± 2.1% |

## Key Features
- XGBoost with SMOTE for class imbalance
- SHAP global + local explainability
- 5-fold cross-validation
- ROC curve + confusion matrix

## How to Run
pip install xgboost shap imbalanced-learn scikit-learn matplotlib seaborn pandas
python heart_disease_prediction.py
