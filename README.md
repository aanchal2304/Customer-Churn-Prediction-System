# Customer Churn Prediction System

A machine learning web application built with **Python** and **Streamlit** that predicts whether a customer is likely to churn based on basic account features such as tenure and monthly charges.

## Live Demo
https://customer-churn-prediction-system-ns3xmqyypo8fab6r9egfaq.streamlit.app/

## Project Overview

Customer churn is one of the most important business problems for subscription-based companies. This project was built to predict customer churn using a machine learning model and provide a simple interactive dashboard for prediction and insight visualization.

The system:
- trains a churn prediction model
- evaluates model performance
- allows user input for live prediction
- shows data insights through charts
- is deployed online using Streamlit Community Cloud

---

## Features

- Predicts whether a customer is likely to churn
- Uses **tenure** and **monthly charges** as input features
- Displays model accuracy
- Provides interactive prediction interface
- Shows visual data insights using charts
- Deployed as a live web application

---

## Tech Stack

- **Python**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Streamlit**
- **Matplotlib**

---

## Machine Learning Workflow

### 1. Data Loading
The dataset is loaded using Pandas.

### 2. Data Cleaning
Missing values are removed and the target column is converted into numeric format:
- `Yes` → `1`
- `No` → `0`

### 3. Feature Selection
The current version uses:
- `tenure`
- `MonthlyCharges`

### 4. Model Training
A **Random Forest Classifier** is trained using `scikit-learn`.

### 5. Evaluation
The model is evaluated using **accuracy score**.

### 6. Deployment
The trained model is saved as `model.pkl` and used in the Streamlit app for real-time predictions.

---

## Project Structure

```bash
Customer-Churn-Prediction-System/
│
├── app.py
├── train_model.py
├── dataset.csv
├── model.pkl
├── requirements.txt
└── README.md
