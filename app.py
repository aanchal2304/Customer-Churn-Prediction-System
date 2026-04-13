import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load dataset (for evaluation)
df = pd.read_csv("dataset.csv")
df = df.dropna()
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

X = df[['tenure', 'MonthlyCharges']]
y = df['Churn']

# Predictions for accuracy
y_pred = model.predict(X)
acc = accuracy_score(y, y_pred)

# UI
st.set_page_config(page_title="Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction System")

st.subheader("Model Performance")
st.metric("Accuracy", f"{acc*100:.2f}%")

# Input section
st.subheader("Predict Customer Churn")

tenure = st.slider("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges", 0.0)

if st.button("Predict"):
    data = np.array([[tenure, monthly_charges]])
    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("⚠️ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")

# Visualization
st.subheader("Data Insights")

st.line_chart(df['MonthlyCharges'])
st.bar_chart(df['tenure'])