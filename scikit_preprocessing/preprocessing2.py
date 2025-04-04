#Quantile Transformer in presprocessing

# ==========================
# 📌 Dataset Scaling & kNN Classification
# ==========================

# 📌 Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline

# ==========================
# 📌 Load the Dataset
# ==========================
df = pd.read_csv('drawndata1.csv')  # Load dataset
print("\n🔹 First 5 rows of the dataset:")
print(df.head())  # Display first 5 rows to understand structure

# ==========================
# 📌 Define Features (X) and Target (y)
# ==========================
# Dataset contains three columns: x, y, and z
# We extract `x` and `y` as features and classify `z`
X = df[['x', 'y']].values  # Convert features to NumPy array
y = (df['z'] == 'a')  # Convert target to binary classification (True/False)

# ==========================
# 📌 Data Visualization (Before Scaling)
# ==========================
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.7)
plt.title("🔹 Raw Data Distribution (Before Scaling)")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.colorbar(label="Class (True = Red, False = Blue)")
plt.show()

# ==========================
# 📌 Apply Data Scaling
# ==========================
# Trying two different scalers: StandardScaler & QuantileTransformer

# ⚡ Option 1: StandardScaler (Mean = 0, Variance = 1)
X_standardized = StandardScaler().fit_transform(X)

# ⚡ Option 2: QuantileTransformer (Uniform Distribution)
X_transformed = QuantileTransformer(n_quantiles=100).fit_transform(X)

# ==========================
# 📌 Data Visualization (After Scaling)
# ==========================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot StandardScaler results
axes[0].scatter(X_standardized[:, 0], X_standardized[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.7)
axes[0].set_title("🔹 StandardScaler Applied")
axes[0].set_xlabel("Feature X (Scaled)")
axes[0].set_ylabel("Feature Y (Scaled)")

# Plot QuantileTransformer results
axes[1].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.7)
axes[1].set_title("🔹 QuantileTransformer Applied")
axes[1].set_xlabel("Feature X (Transformed)")
axes[1].set_ylabel("Feature Y (Transformed)")

plt.colorbar(axes[1].collections[0], ax=axes, location='right', label="Class (True = Red, False = Blue)")
plt.show()

# ==========================
# 📌 Train k-Nearest Neighbors Classifier
# ==========================
# Define kNN model with pipeline
pipeline = Pipeline([
    ('scaler', QuantileTransformer(n_quantiles=100)),  # Best scaling method for this dataset
    ('classifier', KNeighborsClassifier(n_neighbors=5))  # kNN with k=5
])

# Train the model
pipeline.fit(X, y)

# Predict on training data (for visualization)
predictions = pipeline.predict(X)

# ==========================
# 📌 Visualizing kNN Predictions
# ==========================
plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='coolwarm', edgecolor='k', alpha=0.7)
plt.title("🔹 kNN Predictions (k=5, After Scaling)")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.colorbar(label="Predicted Class")
plt.show()

# ==========================
# 📌 Model Performance Evaluation
# ==========================
accuracy = (predictions == y).mean()
print(f"\n✅ Model Accuracy: {accuracy:.2%}")

