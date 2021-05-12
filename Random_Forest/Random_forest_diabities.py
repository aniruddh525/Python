# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:52:01 2021

@author: anirudh.kumar.verma
"""

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# load data
diabities=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\pima-indians-diabetes.data.csv")
diabities.columns
diabities.info()
diabities.head()

diabities.rename(columns={'6':'preg', '148':'plas', '72':'pres', 
                          '35':'skin', '0':'test', '33.6':'mass',
                          '0.627':'pedi', '50':'age', '1':'class'},inplace=True)


diabities_array = diabities.values
X = diabities_array[:,0:8]
Y = diabities_array[:,8]

kfold = KFold(n_splits=10, random_state=7,shuffle=True)
num_trees = 150
max_features = 3
RF_model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
results = cross_val_score(RF_model, X, Y, cv=kfold)
results.mean()






