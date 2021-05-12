# -*- coding: utf-8 -*-
"""
Created on Tue May 11 22:57:25 2021

@author: anirudh.kumar.verma
"""

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


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
seed = 7

# Bagged Decision Trees for Classification

from sklearn.ensemble import BaggingClassifier
kfold = KFold(n_splits=10, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 150
bag_model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
results = cross_val_score(bag_model, X, Y, cv=kfold)
results.mean()


# AdaBoost Classification

from sklearn.ensemble import AdaBoostClassifier

num_trees = 120
seed=7
kfold = KFold(n_splits=10, random_state=seed)
boost_model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
results = cross_val_score(boost_model, X, Y, cv=kfold)
results.mean()


# Stacking Ensemble for Classification

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

kfold = KFold(n_splits=10, random_state=7)


# create the sub models
estimators = []
model1 = LogisticRegression(max_iter=500)
estimators.append(('logistic', model1))
model2 = DecisionTreeClassifier()
estimators.append(('cart', model2))
model3 = SVC()
estimators.append(('svm', model3))

# create the ensemble model
ensemble = VotingClassifier(estimators)
results = cross_val_score(ensemble, X, Y, cv=kfold)
results.mean()














