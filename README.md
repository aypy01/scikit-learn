# 🤖 Scikit-learn Master Notes

Welcome to the core notes of [Scikit-learn Basics](https://scikit-learn.org/0.21/user_guide.html)
. This file serves as a structured, reference-friendly documentation of your entire understanding of Scikit-learn's modeling pipeline — from raw data to final predictions.

---

## 📚 Table of Contents

- [🧠 Basic Workflow](#-1-basic-scikit-learn-workflow)
- [🧪 Preprocessing](#-2-understanding-features-and-preprocessing)
- [📈 Modeling & Evaluation](#-3-modeling-training-and-evaluation)
- [🔢 GridSearchCV & Cross Validation](#-4-gridsearchcv--cross-validation-cv)
- [⚖️ Sample Weights vs Class Weights](#-5-sample-weights-vs-class-weights)
- [🚨 Outlier Detection](#-6-outlier-detection-models)
- [🧮 Precision vs Recall](#-7-precision-vs-recall)
- [🧠 Meta Estimators (Post-processing)](#-8-meta-estimators-post-processing)
- [📦 Summary](#-9-summary)

---

## 🧠 1. Basic Scikit-learn Workflow

> `Data → Model → Fit → Predict → Evaluate`

- **X** = input features (independent variables)
- **y** = target variable (dependent variable)
- **Train** the model using: `model.fit(X, y)`
- **Predict** using: `model.predict(X)`
- Use tools like [Matplotlib](https://matplotlib.org/stable/users/index.html) for visualizing and `metrics` for evaluating.

---

## 🧪 2. Understanding Features and Preprocessing

### ✅ Why Preprocess?

- Feature units vary (e.g., age vs salary).
- Models like **KNN**, **SVM** are scale-sensitive.
- **Scaling** ensures better performance.

### 🔧 Preprocessing Techniques

- `StandardScaler`, `MinMaxScaler` – for normalization
- `PolynomialFeatures` – adds non-linear terms like x², xy, etc.
- `QuantileTransformer` – smooths and reshapes distributions
- `OneHotEncoder` – transforms categorical values into numerical binary columns

> ⚠️ Avoid using `OneHotEncoder` on target `y`. Use with caution and only when suitable.

---

## 📈 3. Modeling: Training and Evaluation

### ⚙️ Standard Code Pipeline

1. Import libraries  
2. Load dataset  
3. Split into X (features) and y (target)  
4. Drop/extract irrelevant columns  
5. Create pipeline (preprocessing + model)  
6. Use `GridSearchCV` for tuning  
7. `model.fit()`  
8. `model.predict()`  
9. Visualization or evaluation using `metrics`  

### ⚠️ Boston Dataset Note

- The `boston` dataset is deprecated due to racial bias. Avoid it.

---

## 🔢 4. GridSearchCV & Cross Validation (CV)

- Automates hyperparameter tuning using cross-validation.
- Evaluates different combinations of parameters.
- Returns the **best-performing model**.
- Improves **generalization** and **accuracy**.

---

## ⚖️ 5. Sample Weights vs Class Weights

| Term            | Purpose                                                   |
|-----------------|-----------------------------------------------------------|
| `class_weight`  | Balances imbalanced classes (e.g., fraud detection)       |
| `sample_weight` | Assigns custom importance to individual training samples  |

---

## 🚨 6. Outlier Detection Models

### 🕵️ Isolation Forest

- **Unsupervised learning** algorithm
- Detects anomalies by randomly isolating observations
- Outliers require fewer splits → easier to isolate

> 🔍 Good for **fraud**, **anomalies**, **rare event detection**

---

## 🧮 7. Precision vs Recall

| Metric     | Meaning                                                                  |
|------------|--------------------------------------------------------------------------|
| Precision  | Among predicted frauds, how many were actually fraud?                    |
| Recall     | Among actual frauds, how many did we catch?                              |

- **Use Precision** when false positives are expensive (e.g., spam detection)
- **Use Recall** when missing real cases is dangerous (e.g., cancer detection)

---

## 🧠 8. Meta Estimators (Post-processing)

These are not actual models — they **enhance or combine** models for better predictions.

| Meta Estimator      | Function                                                         |
|---------------------|------------------------------------------------------------------|
| `VotingClassifier`  | Combines multiple models via majority vote or average            |
| `ThresholdClassifier` | Adjusts classification threshold, useful for imbalanced data  |
| `FeatureUnion`      | Combines multiple feature transformations into one pipeline      |
| `DecayEstimator`    | Weighs recent data more heavily during training                  |
| `GroupedPredictor`  | Trains separate models for different subsets/groups              |

> Note: Although they're called **post-processing**, most of these are applied **before `.predict()`**, as they impact how predictions are made.

---

## 📦 9.Summary
>The overall machine learning process using Scikit-learn follows a logical, modular sequence.
You begin by working with raw data, typically separating it into input features (X) and target labels (y). This data is often inconsistent in scale or format, which is why preprocessing is essential. Steps like scaling (e.g., using StandardScaler), encoding categorical variables (OneHotEncoder), or generating new features (like polynomial ones) help normalize the input for the model.
Once preprocessing is defined, a pipeline can be constructed — bundling both preprocessing and the machine learning model itself. This keeps things clean and consistent across training and inference.
Hyperparameter tuning is done using tools like GridSearchCV, which automatically tests combinations of model settings and uses cross-validation to evaluate which configuration performs best.
Optionally, meta-estimators (like VotingClassifier or threshold adjusters) can be used to further refine the prediction process. These operate on top of the base model and can combine models or adjust decision-making logic.
Finally, the trained model is used for prediction, and performance is measured using evaluation metrics such as accuracy, precision, recall, or F1 score. This full pipeline supports automation, optimization, and scalability of your ML workflow 

---
## 🙌 Connect with Me

If you found this project helpful, or want to connect for collaboration, discussions, or even just to chat about machine learning or 3D metal printing — let’s connect! 
.

  You can find me on: 
  - [Mail](mailto:yadav.aditya595@gmail.com?subject=Hey%20Aditya&body=Just%20saw%20your%20project!)
  - [LinkedIn](https://www.linkedin.com/in/aditya-yadav-77061a33a/)  
  - [Github](https://github.com/Adityeah18)  
  - [YouTube](https://www.youtube.com/@aypy27) 
