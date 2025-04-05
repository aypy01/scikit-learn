# -----------------------------------------
# Fraud Detection using Logistic Regression
# Dataset: creditcard.csv (truncated to 80K rows for demo/training)
# -----------------------------------------

# 📦 Importing Required Libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# 📊 Load and Prepare the Dataset
df = pd.read_csv('creditcard.csv')[:80_000]  # Load first 80,000 rows

# 🎯 Feature Matrix (X) and Target Vector (y)
X = df.drop(columns=['Time', 'Amount', 'Class']).values  # Exclude unscaled or less useful features
y = df['Class'].values  # 0 = Legit, 1 = Fraud

# ⚙️ Logistic Regression Model with Class Weights
# Giving higher weight to minority class (fraud cases) to handle imbalance
model = LogisticRegression(class_weight={0: 1, 1: 2}, max_iter=1000)
model.fit(X, y)

# 🔍 Predict on Training Data
fraud_predictions = model.predict(X).sum()

# 🧾 Output: Total number of predicted fraudulent transactions
print(f"Predicted Fraudulent Transactions: {fraud_predictions}")
