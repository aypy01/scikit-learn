# -----------------------------------------
# Custom Scoring in Logistic Regression (Fraud Detection)
# Metric: Minimum of Precision & Recall to ensure balanced performance
# Dataset: creditcard.csv (truncated to 80K rows)
# -----------------------------------------

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score

# Load the dataset (first 80,000 rows for efficiency)
df = pd.read_csv('creditcard.csv')[:80_000]

# Prepare features and labels
X = df.drop(columns=['Time', 'Amount', 'Class']).values
y = df['Class'].values

# -----------------------------------------
# Custom scoring function
# Returns the lower of precision and recall for a prediction
# This helps ensure neither metric is disproportionately low
# -----------------------------------------
def min_recall_precision(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    return min(recall, precision)

# -----------------------------------------
# GridSearchCV Setup
# - Tune class_weight of class 1 (fraud)
# - Evaluate using precision, recall, and custom min(recall, precision)
# - Refit the model using the custom score
# -----------------------------------------
grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid={'class_weight': [{0: 1, 1: v} for v in np.linspace(1, 20, 10)]},
    scoring={
        'precision': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score),
        'min_both': make_scorer(min_recall_precision)
    },
    refit='min_both',
    return_train_score=True,
    cv=10,
    n_jobs=-1
)

# Run the grid search
grid.fit(X, y)

# -----------------------------------------
# Visualization: All three scores vs class weight
# -----------------------------------------
plt.figure(figsize=(12, 4))
df_results = pd.DataFrame(grid.cv_results_)

for score in ['mean_test_precision', 'mean_test_recall_score', 'mean_test_min_both']:
    plt.plot(
        [cw[1] for cw in df_results['param_class_weight']],
        df_results[score],
        label=score
    )

plt.xlabel('Class Weight for Class 1 (Fraud)')
plt.ylabel('Score')
plt.title('Precision, Recall & min(Precision, Recall) vs Class Weight')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------------------
# Summary:
# - A custom metric like min(precision, recall) helps avoid high precision but poor recall (or vice versa).
# - GridSearchCV gives a flexible way to optimize multiple objectives.
# - Choose class weight where min_both peaks to ensure balanced fraud detection.
# -----------------------------------------
