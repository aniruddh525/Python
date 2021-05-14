# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:54:17 2021

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

forest=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\SVM\\forestfires.csv")
forest.columns
forest.info()
forest.head()
forest.describe()


forest_array = forest.values
X = forest_array[:,2:30]
Y = forest_array[:,30]

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3)

# to check distribution of class
unique, counts = np.unique(y_train, return_counts=True)
dict(zip(unique, counts))

model = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(model,param_grid)
gsv.fit(X_train,y_train)

gsv.best_params_ # {'C': 15, 'gamma': 0.5, 'kernel': 'rbf'}
gsv.best_score_ # 0.753

svm_model = SVC(C= 15, gamma = 50)
svm_model.fit(X_train , y_train)
y_pred = svm_model.predict(X_test)
accuracy_score(y_test, y_pred) * 100 # 70.51
confusion_matrix(y_test, y_pred) 
pd.crosstab(y_test, y_pred)

