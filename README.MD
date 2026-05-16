Cardiovascular Disease Regression: An "Honest" Approach
A straight-to-the-point Machine Learning regression pipeline designed to predict a patient's Systolic Blood Pressure (ap_hi) using purely demographic and lifestyle data.

The "Honest" Methodology
In the raw dataset, both systolic (ap_hi) and diastolic (ap_lo) blood pressures are provided. Predicting systolic pressure while keeping the diastolic reading as a training feature causes target leakage (since they are two halves of the same simultaneous physical measurement).

This script explicitly drops ap_lo, forcing the Random Forest model to genuinely predict blood pressure based only on actual human traits (age, weight, height, cholesterol, and lifestyle).

Requirements
Python 3.x

pandas

scikit-learn

Install the required packages using pip:

Bash
pip install pandas scikit-learn
Setup & Dataset
Download the cardio_train.csv dataset from Kaggle.

Place cardio_train.csv in the exact same directory as your Python script.

Usage
Run the script from your terminal or command prompt:

Bash
python honest_regression.py
Pipeline Overview
When executed, the script automatically performs the following steps:

Data Cleaning: Filters out severe data-entry errors present in the raw Kaggle file (e.g., blood pressures listed in the thousands).

Feature Selection: Drops id, cardio (the classification target), and ap_lo (to prevent cheating).

Training: Splits the data and trains a RandomForestRegressor.

Evaluation: Prints the Mean Absolute Error (MAE), R-squared (R²) Score, and a ranked list of Feature Importances to show which demographic traits drove the predictions.