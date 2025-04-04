# ==============================
# 📌 Polynomial Logistic Regression
# ==============================

# 📌 Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression  # Classification model
from sklearn.preprocessing import PolynomialFeatures  # Generates polynomial features
from sklearn.pipeline import Pipeline  # Combines preprocessing and modeling steps

# ==============================
# 📌 Load Dataset
# ==============================
df = pd.read_csv('drawndata2.csv')  # Load dataset
print("\n🔹 First 5 rows of the dataset:")
print(df.head())  # Display first 5 rows to understand structure

# ==============================
# 📌 Define Features (X) and Target (y)
# ==============================
X = df[['x', 'y']].values  # Selecting 'x' and 'y' columns as input features
y = (df['z'] == 'a')  # Convert target to binary classification (True for 'a', False otherwise)

# ==============================
# 📌 Create a Machine Learning Pipeline
# ==============================
# Steps:
# 1. Apply PolynomialFeature transformation (degree=2)
# 2. Train Logistic Regression on the transformed features
pipe = Pipeline([
    ('poly_features', PolynomialFeatures(degree=2)),  # Expands features to include x², y², and xy terms
    ('logistic_model', LogisticRegression())  # Classification model
])

# Train the model and make predictions
prediction = pipe.fit(X, y).predict(X)

# ==============================
# 📌 Visualizing the Classification Result
# ==============================
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=prediction, cmap='coolwarm', edgecolor='k', alpha=0.7)
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.title("🔹 Logistic Regression with Polynomial Features")
plt.colorbar(label="Predicted Class")
plt.show()

# ==============================
# 📌 Save the Plot for Documentation
# ==============================
plt.savefig("logistic_regression_plot.png", dpi=300)

# ==============================
# 📌 Print Model Accuracy
# ==============================
accuracy = (prediction == y).mean()
print(f"\n✅ Model Accuracy: {accuracy:.2%}")
