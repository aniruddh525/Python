# -*- coding: utf-8 -*-
"""
Created on Thu May 13 11:22:12 2021

@author: anirudh.kumar.verma
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# load data
diabities=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\pima-indians-diabetes.data.csv")
diabities.columns
diabities.info()
diabities.head()
diabities.describe()

diabities.rename(columns={'6':'preg', '148':'plas', '72':'pres', 
                          '35':'skin', '0':'test', '33.6':'mass',
                          '0.627':'pedi', '50':'age', '1':'class'},inplace=True)


diabities_array = diabities.values
X = diabities_array[:,0:8]
Y = diabities_array[:,8]

num_folds = 10
kfold = KFold(n_splits=10)

KNN_model = KNeighborsClassifier(n_neighbors=17)
results = cross_val_score(KNN_model, X, Y, cv=kfold)

print(results.mean()) # 0.75372

# Grid Search for Algorithm Tuning

from sklearn.model_selection import GridSearchCV

n_neighbors = np.array(range(1,40))
param_grid = dict(n_neighbors=n_neighbors)


Grid_model = KNeighborsClassifier()
grid = GridSearchCV(estimator=Grid_model, param_grid=param_grid)
grid.fit(X, Y)


print(grid.best_score_) # 0.7588
print(grid.best_params_) # 14


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


