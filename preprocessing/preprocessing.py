#Basic Pandas

# =======================================
# 📌 1. Import required libraries
# =======================================
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# =======================================
# 📌 2. Load dataset
# =======================================
df = pd.read_csv('drawndata1.csv')  # Reads the CSV file into a DataFrame

# =======================================
# 📌 3. Preview dataset
# =======================================
print(df.head(3))  # Shows first 3 rows

# =======================================
# 📌 4. Define X (features) and y (target)
# =======================================

X = df[['x', 'y']].values  # Extracts columns 'x' and 'y' as a NumPy array
y = df['z'] == 'a'  # Creates a Boolean array (True if 'z' == 'a', False otherwise)

# =======================================
# 📌 5. Plot the scatter plot
# =======================================

plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.5)  # Color based on 'y', make points semi-transparent
plt.xlabel("X-axis values")
plt.ylabel("Y-axis values")
plt.title("Scatter Plot of X vs Y, Colored by Class")
plt.axis('equal')  # Ensures proper scaling
plt.show()
