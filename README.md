# Scikit-learn

This repository contains structured notes on Scikit-learn's machine learning workflow, from preprocessing raw data to applying meta estimators for enhanced predictions. It is intended as a concise, reference-friendly guide for anyone learning or refining their skills in Scikit-learn 
and a project [Heart Disease Predictor](https://github.com/aypy01/scikit-learn/blob/main/heart_disease_predictor.ipynb)

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
- [Heart Disease Predictor](https://github.com/aypy01/scikit-learn/blob/main/heart_disease_predictor.ipynb)
- [Summary](#summary)
- [Author](#author)
- [License](#license)

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
- Unrealistic/extreme inputs cause unstable predictions; realistic values give consistent results.
Inconsistent outputs were due to random splits without a fixed seed.

- Dataset imbalance needs stratified splitting or resampling.

- Saving models with pickle ensures correct loading and reuse.

- Preprocessing (e.g., scaling) generally improves stability and performance.

- Achieved accuracy 98% without overfitting.

- Balanced data, realistic inputs, reproducibility, and preprocessing are key for reliable ML pipelines.

---

## Thank You
Knowledge should not be gated behind paywalls or exclusivity. This repository exists so that anyone can access structured, practical Scikit-learn notes without restriction.  
The journey doesn’t end here. After mastering meta estimators, take the next step with the full-fledged Scikit-learn project 

[am_i_cooked](https://github.com/aypy01/am-i-cooked) A more advanced heart disease predictor built on a larger dataset, featuring a Flask-based web UI for deployment.

## Author
 <p align="left">
  Created and maintained by
  <a href="https://github.com/aypy01" target="_blank"> Aaditya Yadav</a>
  <a href="https://github.com/aypy01" target="_blank">
    <img src="https://img.shields.io/badge/aypy01-000000?style=flat-square&logo=github&logoColor=00FF80" alt="GitHub Badge"/>
  </a>
</p>

</p>
<p align="left">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&duration=3000&pause=500&color=00FF80&center=false&vCenter=false&width=440&lines=Break+Things+First%2C+Understand+Later;Built+to+Debug%2C+Not+Repeat;Learning+What+Actually+Sticks;Code.+Observe.+Refine." alt="Typing SVG" />
</p>

## License

This project is licensed under the [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT).


