# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:44:57 2021

@author: anirudh.kumar.verma
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score

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

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3)

# to check distribution of class
unique, counts = np.unique(y_test, return_counts=True)
dict(zip(unique, counts))

model = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(model,param_grid)
gsv.fit(X_train,y_train)

gsv.best_params_
gsv.best_score_ 

svm_model = SVC(C= 15, gamma = 50)
svm_model.fit(X_train , y_train)
y_pred = svm_model.predict(X_test)
accuracy_score(y_test, y_pred) * 100
confusion_matrix(y_test, y_pred)
pd.crosstab(y_test, y_pred)















