# -----------------------------------------
# Thresholding in Classification Models
# Using Thresholder (from scikit-lego) to manually control decision threshold
# -----------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, accuracy_score, recall_score

from sklego.meta import Thresholder

# -----------------------------------------
# Why use make_blobs instead of make_classification?
# - make_blobs generates clearly separated Gaussian clusters
# - Better control over cluster centers and spread (via std dev)
# - Useful for **visualizing** threshold effects in a clean, linear setup
# - make_classification has more complexity (irrelevant for threshold demo)
# -----------------------------------------

X, y = make_blobs(
    n_samples=2000,                     # Total samples
    centers=[(0, 0), (1.5, 1.5)],       # Two class centers
    cluster_std=[1, 0.5]                # Different spread for each class
)

# -----------------------------------------
# Plotting the synthetic data
# -----------------------------------------
plt.scatter(X[:, 0], X[:, 1], c=y, s=5)
plt.title("Synthetic Classification Dataset using make_blobs")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
