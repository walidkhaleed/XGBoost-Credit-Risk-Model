# 🏦 XGBoost Credit Risk Model & Explainable AI

![SHAP Summary Plot](shap_summary.png)

## Project Overview
This project focuses on building a production-grade machine learning pipeline to predict loan defaults. Utilizing **XGBoost**, the model is specifically tuned to maximize Recall for the default class, ensuring the financial institution minimizes catastrophic losses from bad loans. 

To satisfy financial regulatory requirements for algorithmic transparency, the "black box" model is decoded using **SHAP (SHapley Additive exPlanations)**, proving the exact mathematical drivers behind every loan approval and denial.

## Core Machine Learning Challenges Handled
* **Sparsity-Aware Split Finding:** Bypassed traditional mean/median imputation. Identified hardcoded data errors (`99999999`) and converted them to `NaN`, allowing XGBoost's native missing-value algorithms to find the optimal split path naturally.
* **Imbalanced Classification:** Instead of generating synthetic data (SMOTE), the class imbalance was handled mathematically using XGBoost's `scale_pos_weight` parameter, heavily penalizing the algorithm for missing true defaults.
* **Overfitting Prevention:** Implemented `GridSearchCV` for hyperparameter tuning paired with `early_stopping_rounds`, automatically halting tree-building the moment validation scores plateaued.

## Methodology
* **Algorithm:** Extreme Gradient Boosting (`XGBClassifier`).
* **Evaluation Metrics:** Discarded overall accuracy in favor of the **Confusion Matrix and Class 1 Recall**, prioritizing the identification of high-risk customers.
* **Explainable AI:** Generated SHAP summary plots indicating that years of seniority, total income, and debt-to-income ratios were the strongest driving forces behind the model's logic.

## Files in this Repository
* `Credit_Risk_XGBoost.ipynb`: The core Jupyter Notebook containing the data pipeline, GridSearch tuning, model training, and SHAP visualizations.
* `shap_summary.png`: The generated visualization decoding the XGBoost decision-making process.
