# -------------------------------------------------------
# DECAY: Useful for Time-Series with Uneven Importance
# (e.g. newer data more relevant than older)
# -------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from sklearn.dummy import DummyRegressor
from sklego.datasets import make_simpleseries
from sklego.meta import GroupedPredictor

# -------------------------------------------------------
# Generate Dummy Time-Series Data
# - yt: target variable
# - date: timeline
# - m: extracted month as group
# -------------------------------------------------------
yt = make_simpleseries(seed=1)
dates = pd.date_range("2000-01-01", periods=len(yt))
df = (
    pd.DataFrame({"yt": yt, "date": dates})
    .assign(m=lambda d: d.date.dt.month)  # extracting month for grouping
    .reset_index()
)

# -------------------------------------------------------
# GroupedPredictor:
# - Wraps a base model (DummyRegressor here)
# - Trains separate models for each group (month in this case)
# - Think of it like: 12 tiny models for Jan to Dec
# -------------------------------------------------------
model = GroupedPredictor(DummyRegressor(), groups=['m'])

# Fit the grouped model
model.fit(df[['m']], df['yt'])

# -------------------------------------------------------
# Plot Actual vs Predicted
# -------------------------------------------------------
plt.figure(figsize=(12, 3))
plt.plot(df['yt'], alpha=0.5, label='Actual')                     # Original signal
plt.plot(model.predict(df[['m']]), label='Grouped Prediction')   # Grouped prediction
plt.legend()
plt.title("Grouped Predictor (by Month)")
plt.xlabel("Time Index")
plt.ylabel("yt")
plt.grid(True)
plt.show()
