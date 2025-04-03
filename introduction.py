# =======================================
# 📌 Import necessary libraries
# =======================================
from sklearn.datasets import fetch_california_housing, load_wine
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd

##Initilal Trial
# =======================================
# 📌 Load the California Housing Dataset
# =======================================
housing = fetch_california_housing()
print(
    housing.data.shape, housing.target.shape
)  # Prints the shape of features and target
print(housing.feature_names[:6])  # Prints the first 6 feature names

# 2nd Program
# =======================================
# 📌 Load the Wine Dataset
# =======================================
X, y = load_wine(return_X_y=True)  # X contains features, y contains target labels

# =======================================
# 📌 Initialize and Train a Model
# =======================================

# Using KNeighborsRegressor (Can be switched to LinearRegression)
model = KNeighborsRegressor()
# model = LinearRegression()  # Uncomment this line to use Linear Regression instead

# Train (fit) the model using the Wine dataset
model.fit(
    X, y
)  # .fit step is the the 1st thing as hirarchy only then the predict will run

# Make predictions on the same dataset
predictions = model.predict(X)[:3]  # Display only first 3 predictions for clarity
print(predictions)

# Program 3
# =======================================
# 📌 Simple Visualization for Regression
# =======================================

# Load California Housing Dataset again
X, y = fetch_california_housing(return_X_y=True)

# Train & predict using Linear Regression
model = LinearRegression().fit(X, y)
predictions = model.predict(X)

# Plot predictions vs. actual values
plt.scatter(predictions, y)
plt.show()

# Program 4
# =======================================
# 📌 Concept of Pipelines in Machine Learning
# =======================================

"""
A Pipeline combines multiple steps into one object.
It helps in structuring the ML workflow by automatically applying preprocessing 
before training the model.
"""

X, y = fetch_california_housing(return_X_y=True)

# Pipeline with Scaling + KNN Regression (n_neighbors = 1)
pipe = Pipeline(
    [
        ("scale", StandardScaler()),  # Standardizes the data
        (
            "model",
            KNeighborsRegressor(n_neighbors=1),
        ),  # Applies kNN Regression with 1 neighbor
    ]
)

# Note: Using `n_neighbors=1` is risky as it memorizes training data (overfitting risk)

# Train & predict
predictions = pipe.fit(X, y).predict(X)

# Plot predictions vs actual values
plt.scatter(predictions, y)
plt.show()

# Program 4
# =======================================
# 📌 Grid Search with Cross-Validation
# =======================================

"""
GridSearchCV is used to find the best hyperparameters for a model.
It tests multiple values and selects the one with the highest performance.

- param_grid: Defines which hyperparameters to tune
- cv=3: Performs 3-fold cross-validation
"""

model = GridSearchCV(
    estimator=pipe,  # Uses the previously defined pipeline
    param_grid={
        "model__n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    },  # Testing k values from 1 to 10
    cv=3,  # Cross-validation with 3 folds (splits dataset into 3 parts)
)

# Fit the model using GridSearchCV
model.fit(X, y)

print(pd.DataFrame(model.cv_results_))


# =======================================
# 📌 Best Parameters Selection
# =======================================

"""
- The rank_test_score column shows which parameter gave the best results.
- The best parameters can be directly accessed using `model.best_params_`
"""

best_params = model.best_params_
print(f"Best n_neighbors: {best_params['model__n_neighbors']}")

# Train the final model with the best found hyperparameter
best_knn_model = KNeighborsRegressor(n_neighbors=best_params["model__n_neighbors"]).fit(
    X, y
)
final_predictions = best_knn_model.predict(X)

# Plot the final model's performance
plt.scatter(final_predictions, y)
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.title(
    f"KNN Best Model (n={best_params['model__n_neighbors']}) Predictions vs Actual"
)
plt.show()

