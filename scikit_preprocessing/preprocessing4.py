# ==============================
# 📌 One-Hot Encoding with Scikit-Learn
# ==============================

# 📌 Import necessary libraries
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# ==============================
# 📌 Define Categorical Data
# ==============================
# We have ordinal categories: "low", "medium", "high"
# Reshaping (-1,1) to ensure it's in column format for encoding
arr = np.array(["low", "low", "medium", "high"]).reshape(-1, 1)
print("\n🔹 Original Categorical Data:\n", arr)

# ==============================
# 📌 Initialize OneHotEncoder
# ==============================
# `sparse_output=False` ensures the output is a dense NumPy array instead of a sparse matrix.
# `handle_unknown='ignore'` prevents errors when an unknown category appears during transformation.
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

# ==============================
# 📌 Fit and Transform Data
# ==============================
# - The `fit()` method learns unique categories from the data.
# - The `transform()` method converts categorical data into one-hot encoded vectors.
encoded_arr = encoder.fit_transform(arr)

print("\n🔹 One-Hot Encoded Output:\n", encoded_arr)

# ==============================
# 📌 Handling Unknown Categories
# ==============================
# The category "zero" was not present in the original dataset.
# Since `handle_unknown='ignore'`, it doesn't throw an error but returns an all-zero vector.
unknown_category = np.array([["zero"]])
encoded_unknown = encoder.transform(unknown_category)

print("\n🔹 Encoding for an Unknown Category ('zero'):\n", encoded_unknown)
