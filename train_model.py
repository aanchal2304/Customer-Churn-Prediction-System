import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv("dataset.csv")

# Basic cleaning
df = df.dropna()

# Convert target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Select features (simple)
X = df[['tenure', 'MonthlyCharges']]
y = df['Churn']

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved")