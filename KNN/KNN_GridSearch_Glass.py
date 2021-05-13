# -*- coding: utf-8 -*-
"""
Created on Thu May 13 14:42:03 2021

@author: anirudh.kumar.verma
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# load data
glass=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\KNN\\glass.csv")
glass.columns
glass.info()
glass.head()
glass.describe()
glass.shape

glass_array = glass.values
X = glass_array[:,0:9]
Y = glass_array[:,9]

# to check distribution of class
unique, counts = np.unique(Y, return_counts=True)
dict(zip(unique, counts))

num_folds = 10
kfold = KFold(n_splits=10)

KNN_model = KNeighborsClassifier(n_neighbors=17)
results = cross_val_score(KNN_model, X, Y, cv=kfold)

print(results.mean()) # 0.43

# Grid Search for Algorithm Tuning

from sklearn.model_selection import GridSearchCV

n_neighbors = np.array(range(1,40))
param_grid = dict(n_neighbors=n_neighbors)

Grid_model = KNeighborsClassifier()
grid = GridSearchCV(estimator=Grid_model, param_grid=param_grid)
grid.fit(X, Y)


print(grid.best_score_) # 0.64
print(grid.best_params_) # 1


# Visualizing the CV results

import matplotlib.pyplot as plt 

# choose k between 1 to 41
k_range = range(1, 41)
k_scores = []
# use iteration to caclulator different k in models, then return the average accuracy based on the cross validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=5)
    k_scores.append(scores.mean())
# plot to see clearly
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.show()

