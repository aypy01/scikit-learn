import  numpy as np
import pandas as pd
import matplotlib.pylab as plt

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score,precision_score,recall_score,make_scorer

from sklego.meta import Thresholder

X,y=make_blobs(n_samples=1000,centers=[(0,0),(1.5,1.5)],cluster_std=[1,0.5])

#Threshold Parameter
m1=Thresholder(model=LogisticRegression(solver='lbfgs'),threshold=0.1).fit(X,y) #solver lbfs=finds the best weight
m2=Thresholder(model=LogisticRegression(solver='lbfgs'),threshold=0.9).fit(X,y)

#Matplotlib
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], c=y, s=5) #This is for the whole plot scattred way ,whole data without the prediction
plt.title("original data")
plt.subplot(132)
plt.scatter(X[:, 0], X[:, 1], c=m1.predict(X), s=5)#Is the predicting part also happening here with plot making. i mean i usually do seperate line here for predicting # This is plot showing only when m=0.1 and predictin X
plt.title("threshold=0.1")
plt.subplot(133)
plt.scatter(X[:, 0], X[:, 1], c=m2.predict(X), s=5) #This is plot showing only when m=0.1 and predictin X
plt.title("threshold=0.9")
plt.show()
