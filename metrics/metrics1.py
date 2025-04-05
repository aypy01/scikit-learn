# ==========================
# 📌 Import Necessary Libraries
# ==========================
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

# ==========================
# 📌 Load the Dataset
# ==========================
df = pd.read_csv('creditcard.csv')[:80_000]  # Selecting only the first 80,000 rows for faster processing

# ==========================
# 📌 Feature Selection (X) & Target Variable (y)
# ==========================
# 🔹 Removing unnecessary columns:
#    - 'Time' → Timestamp (not useful for fraud detection)
#    - 'Amount' → Transaction amount (could be useful, but excluding for now)
#    - 'Class' → Target label (should not be in X)
X = df.drop(columns=['Time', 'Amount', 'Class']).values  # Convert to NumPy array for ML models

# 🔹 Extracting the target variable ('Class')
#    - 0 → Legitimate Transaction
#    - 1 → Fraudulent Transaction
y = df['Class'].values

# ==========================
# 📌 Dataset Summary
# ==========================
fraud_cases = y.sum()  # Count total fraud cases in dataset

print(f"✅ X Shape: {X.shape} | y Shape: {y.shape}")
print(f"🚨 Fraud Cases: {fraud_cases} out of {len(y)} transactions ({(fraud_cases / len(y)) * 100:.4f}%)")
