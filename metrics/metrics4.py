# -----------------------------------------
# Tuning Logistic Regression for Fraud Detection
# Visualizing the effect of class weight on Precision and Recall
# Dataset: creditcard.csv (80,000 samples)
# -----------------------------------------

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score

# Load the dataset (first 80,000 rows)
df = pd.read_csv('creditcard.csv')[:80_000]

# Prepare features and target variable
X = df.drop(columns=['Time', 'Amount', 'Class']).values
y = df['Class'].values

# Setup GridSearchCV to find optimal class_weight for class 1 (fraud)
grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=1000),
    param_grid={'class_weight': [{0: 1, 1: v} for v in np.linspace(1, 20, 10)]},  # Trying weights from 1 to 20 (10 steps)
    scoring={
        'precision': make_scorer(precision_score),
        'recall_score': make_scorer(recall_score)
    },
    refit='precision',           # Optimize for precision
    return_train_score=True,     # Track training score
    cv=10,                       # 10-fold cross-validation
    n_jobs=-1                    # Use all cores
)

# Train the model
grid.fit(X, y)

# -----------------------------------------
# Visualization: Precision and Recall vs Class Weight
# -----------------------------------------
plt.figure(figsize=(12, 4))
df_results = pd.DataFrame(grid.cv_results_)

# Plotting the scores for each class weight tried
for score in ['mean_test_precision', 'mean_test_recall_score']:
    plt.plot(
        [cw[1] for cw in df_results['param_class_weight']],
        df_results[score],
        label=score
    )

plt.xlabel('Class Weight for Class 1 (Fraud)')
plt.ylabel('Score')
plt.title('Precision and Recall vs Class Weight')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------------------
# Notes:
# - GridSearchCV is flexible and allows custom scorers for optimization.
# - Default scoring in LogisticRegression is accuracy, which is not ideal for imbalanced datasets.
# - Here, precision and recall are visualized to help strike a balance between false positives and false negatives.
# - Where the two lines intersect might indicate a good trade-off point.
# -----------------------------------------
