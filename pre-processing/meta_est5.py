# -------------------------------------------------------
# Threshold Tuning using Thresholder (sklego) + GridSearchCV
# -------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score, precision_score, recall_score, make_scorer

from sklego.meta import Thresholder  # Used to manually set classification threshold

# -------------------------------------------------------
# Generate Synthetic Dataset using make_blobs
# - Two class centers
# - Different cluster spread (std dev)
# -------------------------------------------------------
X, y = make_blobs(
    n_samples=1000,
    centers=[(0, 0), (1.5, 1.5)],
    cluster_std=[1, 0.5]
)

# -------------------------------------------------------
# Define Pipeline with Thresholder (initial threshold=0.1)
# - You can change the threshold dynamically using GridSearchCV
# - Model here is LogisticRegression (linear classifier)
# -------------------------------------------------------
pipe = Pipeline([
    ('model', Thresholder(LogisticRegression(solver='lbfgs'), threshold=0.1))
])

# -------------------------------------------------------
# GridSearchCV setup to tune threshold
# - Searches over threshold range [0.1, 0.9]
# - Evaluates with precision, accuracy, and recall
# - Refit based on precision
# - 10-fold CV
# -------------------------------------------------------
model = GridSearchCV(
    estimator=pipe,
    param_grid={'model__threshold': np.linspace(0.1, 0.9, 30)},
    scoring={
        'precision_score': make_scorer(precision_score),
        'accuracy_score': make_scorer(accuracy_score),
        'recall_score': make_scorer(recall_score)
    },
    refit='precision_score',
    cv=10
)

# -------------------------------------------------------
# Fit model on data, then predict and visualize
# -------------------------------------------------------
model.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=model.predict(X), s=5)
plt.title("Prediction with Optimized Threshold")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()
