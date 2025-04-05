# -----------------------------------------
# Fraud Detection using Logistic Regression
# Dataset: creditcard.csv (truncated to 80K rows for demo/training)
# -----------------------------------------

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Load the dataset (first 80,000 samples)
df = pd.read_csv('creditcard.csv')[:80_000]

# Prepare features and target
X = df.drop(columns=["Time", "Amount", "Class"]).values
y = df["Class"].values

# GridSearchCV for tuning class_weight of Logistic Regression
grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid={'class_weight': [{0: 1, 1: v} for v in range(1, 4)]},  # Trying different weights for class 1 (fraud)
    cv=4,                # 4-fold cross-validation
    n_jobs=-1            # Use all CPU cores
)

# Train the model and predict on the same data (demo purposes only)
grid.fit(X, y)
predicted_frauds = grid.predict(X).sum()
print("Total predicted frauds:", predicted_frauds)

# Display cross-validation results
print(pd.DataFrame(grid.cv_results_))

# -----------------------------------------
# Notes:
# - The default scoring for LogisticRegression is accuracy.
# - Accuracy may be misleading for imbalanced datasets like fraud detection.
# - GridSearchCV can be extended with custom scorers (precision, recall, F1).
# - This setup helps tune model sensitivity by adjusting class weights.
# -----------------------------------------
