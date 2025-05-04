# --------------------------------------------
# Program 1: Visualizing a Synthetic Dataset for Classification
# --------------------------------------------

# Meta-estimators are higher-level models that use other models internally.
# For example, VotingClassifier, BaggingClassifier, GridSearchCV, etc.
# These can:
# - Combine multiple models (e.g., run them in parallel)
# - Control their hyperparameters
# - Post-process outputs (like averaging probabilities, majority voting)

# --------------------------------------------
# 📦 Importing required libraries
# --------------------------------------------

import numpy as np
import matplotlib.pylab as plt
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier

# --------------------------------------------
# 🧪 Generating a toy dataset using `make_classification`
# --------------------------------------------

X, y = make_classification(
    n_samples=2000,        # Total 2000 data points
    n_features=2,          # Only 2 features (good for 2D plotting)
    n_redundant=0,         # No redundant features
    class_sep=1.75,        # Controls how separable the classes are
    flip_y=0.1,            # 10% noise — randomly flip labels
    random_state=21        # Ensures reproducibility
)

# Note: `make_classification` supports many more parameters for more complex data generation

# --------------------------------------------
# 🖼️ Visualizing the dataset
# --------------------------------------------

# `c=y` means color each point based on its class (label)
# `s=5` sets the marker size to 5 (small dots)
plt.scatter(X[:, 0], X[:, 1], c=y, s=5)
plt.title("Synthetic Binary Classification Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
