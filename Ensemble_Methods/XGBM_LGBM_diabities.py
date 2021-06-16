# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:34:33 2021

@author: anirudh.kumar.verma
"""

# First XGBoost model for Pima Indians dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# load data
dataset = loadtxt("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\pima-indians-diabetes.data.csv", delimiter=",")
# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]


# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)


# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)


# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]


# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

predictions

### change parameter

model_1 = XGBClassifier(n_jobs=4,n_estimators=1000,learning_rate=0.05)
model_1.fit(X_train, y_train)
y_pred = model_1.predict(X_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


##Light GBM***

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
dataset = pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\pima-indians-diabetes.data.csv")
# split data into X and y
X = dataset.iloc[:,0:8]
Y = dataset.iloc[:,8]


# Splitting the dataset into the Training set and Test set

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


import lightgbm as lgb
d_train = lgb.Dataset(x_train, label=y_train)

params = {}
params['learning_rate'] = 0.003
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
params['sub_feature'] = 0.5
params['num_leaves'] = 10
params['min_data'] = 50
params['max_depth'] = 10

clf = lgb.train(params, d_train, 200)

#Prediction
y_pred=clf.predict(x_test)

predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)

accuracy
