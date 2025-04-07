import numpy as np
import matplotlib.pylab as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier

# -----------------------------------------
# Creating synthetic classification dataset
# -----------------------------------------
X, y = make_classification(
    n_samples=2000,           # Total data points
    n_features=2,             # Only 2 features (for 2D visualization)
    n_redundant=0,            # No redundant (useless) features
    class_sep=1.75,           # Controls how well-separated the classes are
    flip_y=0.1,               # 10% of labels randomly flipped (adds noise)
    random_state=21           # Ensures reproducibility
)
def make_plots():
    X_new = np.concatenate([
        np.random.uniform(np.min(X[:, 0]), np.max(X[:, 0]), (20000, 1)),
        np.random.uniform(np.min(X[:, 1]), np.max(X[:, 1]), (20000, 1))
    ], axis=1)
    plt.figure(figsize=(16, 4))
    plt.subplot(141)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=5)
    plt.title("original data")
    plt.subplot(142)
    plt.scatter(X_new[:, 0], X_new[:, 1], c=clf1.predict_proba(X_new)[:, 1], s=5)
    plt.title("ens1")
    plt.subplot(143)
    plt.scatter(X_new[:, 0], X_new[:, 1], c=clf2.predict_proba(X_new)[:, 1], s=5)
    plt.title("ens2")
    plt.subplot(144)
    plt.scatter(X_new[:, 0], X_new[:, 1], c=clf3.predict_proba(X_new)[:, 1], s=5)
    plt.title("ens3")

plt.scatter(X[:, 0], X[:, 1], c=y, s=5)

# -----------------------------------------
# Classifier 1: Logistic Regression
# -----------------------------------------
clf1 = LogisticRegression().fit(X, y)

# -----------------------------------------
# Classifier 2: K-Nearest Neighbors
# -----------------------------------------
clf2 = KNeighborsClassifier(n_neighbors=5).fit(X, y)

# -----------------------------------------
# Classifier 3: Voting Classifier (Ensemble)
# Combines clf1 and clf2 predictions
# voting='soft' means it averages predicted probabilities instead of labels
# weights=[10.5, 2.5] gives more importance to clf1 (LogReg) in the voting
# -----------------------------------------
clf3 = VotingClassifier(
    estimators=[('clf1', clf1), ('clf2', clf2)],
    voting='soft',
    weights=[10.5, 2.5]
)
clf3.fit(X, y)

# -----------------------------------------
# Generate visualization
# -----------------------------------------
make_plots()
plt.show()
