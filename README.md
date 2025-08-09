# Scikit-learn

This repository contains structured notes on Scikit-learn's machine learning workflow, from preprocessing raw data to applying meta estimators for enhanced predictions. It is intended as a concise, reference-friendly guide for anyone learning or refining their skills in Scikit-learn.

---

## Table of Contents
- [Basic Workflow](#basic-workflow)
- [Preprocessing](#preprocessing)
- [Modeling and Evaluation](#modeling-and-evaluation)
- [GridSearchCV and Cross Validation](#gridsearchcv-and-cross-validation)
- [Sample Weights vs Class Weights](#sample-weights-vs-class-weights)
- [Outlier Detection](#outlier-detection)
- [Precision vs Recall](#precision-vs-recall)
- [Meta Estimators](#meta-estimators)
- [Summary](#summary)

---

## Basic Workflow
Data → Model → Fit → Predict → Evaluate  
The standard process involves defining input features (**X**), target labels (**y**), fitting a model, making predictions, and evaluating results using metrics.

---

## Preprocessing
Preprocessing ensures data consistency and better model performance.  
Key tools:
- `StandardScaler`, `MinMaxScaler` for normalization
- `PolynomialFeatures` for non-linear terms
- `QuantileTransformer` for reshaping distributions
- `OneHotEncoder` for categorical encoding

---

## Modeling and Evaluation
Steps:
1. Import libraries and load data
2. Separate into X and y
3. Preprocess and build pipelines
4. Train and tune models
5. Evaluate with metrics and visualization

Note: Avoid deprecated datasets such as Boston due to bias concerns.

---

## GridSearchCV and Cross Validation
`GridSearchCV` automates hyperparameter tuning and improves generalization by testing multiple configurations through cross-validation.

---

## Sample Weights vs Class Weights
| Term            | Purpose                                                   |
|-----------------|-----------------------------------------------------------|
| `class_weight`  | Balances imbalanced target classes                        |
| `sample_weight` | Assigns custom importance to specific samples             |

---

## Outlier Detection
Example: **Isolation Forest**  
Unsupervised anomaly detection by isolating observations. Outliers are identified as points requiring fewer splits.

---

## Precision vs Recall
| Metric     | Meaning                                           |
|------------|---------------------------------------------------|
| Precision  | Of predicted positives, how many were correct     |
| Recall     | Of actual positives, how many were identified     |

---

## Meta Estimators
Meta estimators enhance or combine models:
- `VotingClassifier` for model aggregation
- Threshold adjusters for classification control
- `FeatureUnion` for combining feature transformations
- Group-based predictors for segmented training

---

## Summary
Scikit-learn workflows follow a modular approach: preprocess data, define pipelines, tune with GridSearchCV, and optionally apply meta estimators for improved performance. The process encourages reproducibility, scalability, and clarity in machine learning projects.

---

## Thank You
Knowledge should not be gated behind paywalls or exclusivity. This repository exists so that anyone can access structured, practical Scikit-learn notes without restriction.  
The journey does not end here after mastering meta estimators, explore the Scikit-learn project `am_i_cooked` for the next step.

