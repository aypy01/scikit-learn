import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from sklearn.dummy import DummyRegressor
from sklego.meta import DecayEstimator, GroupedPredictor
from sklego.datasets import make_simpleseries

from sklego.meta.decay_estimator import exponential_decay

# Generate a simple time-series using sklego's helper — yt is the target
yt = make_simpleseries(seed=1)

# Create a datetime index and structure the dataset
dates = pd.date_range("2000-01-01", periods=len(yt))
df = (pd.DataFrame({"yt": yt,
                    "date": dates})
        .assign(m=lambda d: d.date.dt.month)  # Extract month as group label
        .reset_index())  # Keep index for ordering/time-awareness

# Model 1 — no decay applied, simple GroupedPredictor using month-based grouping
model1 = (GroupedPredictor(DummyRegressor(), groups=["m"]))
model1.fit(df[['m']], df['yt'])

# Model 2 — same as above but now with decay logic applied
# This uses DecayEstimator to apply exponentially decreasing weights (decay) within groups
# decay_func='exponential' tells the estimator to use exponential decay (built-in string ref)
# decay_kwargs={'decay_rate': 0.9} controls the decay curve — closer to 1 means slower decay (i.e., newer and older points are weighted more equally)
# Lower values like 0.5 would emphasize newer data more heavily
model2 = (GroupedPredictor(
            DecayEstimator(DummyRegressor(),
                           decay_func='exponential',
                           decay_kwargs={'decay_rate': 0.9}),
            groups=['m']))  # Grouping still by month
model2.fit(df[['m']], df['yt'])  # Even though we predict with index later, fitting still just needs group info

# Visualization
plt.figure(figsize=(12, 3))
plt.plot(df['yt'], alpha=0.5)  # Actual time-series (ground truth)
plt.plot(model1.predict(df[['m']]), label="grouped")  # Prediction without decay
plt.plot(model2.predict(df[['index', 'm']]), label="decayed")  # Prediction with decay applied
plt.legend()
plt.show()
