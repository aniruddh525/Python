# -*- coding: utf-8 -*-
"""
Created on Fri May 14 12:13:16 2021

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

salary=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\SVM\\SalaryData_Train(1).csv")
salary.columns
salary.info()
salary.head()
salary.describe()

dummies=pd.get_dummies(salary)
dummies.columns
dummies=dummies.drop({'Salary_ <=50K','Salary_ >50K'},axis=1)
final_salary=pd.concat([dummies,salary['Salary']],axis=1)

salary_array = final_salary.values
X = salary_array[0:10000,0:102]
Y = salary_array[0:10000,102]

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.3)

# to check distribution of class
unique, counts = np.unique(y_train, return_counts=True)
dict(zip(unique, counts))

model = SVC()
param_grid = [{'kernel':['rbf'],'gamma':[50,5,10,0.5],'C':[15,14,13,12,11,10,0.1,0.001] }]
gsv = GridSearchCV(model,param_grid)
gsv.fit(X_train,y_train)

gsv.best_params_ # {'C': 15, 'gamma': 50, 'kernel': 'rbf'}
gsv.best_score_ # 0.7857

svm_model = SVC(C= 15, gamma = 50)
svm_model.fit(X_train , y_train)
y_pred = svm_model.predict(X_test)
accuracy_score(y_test, y_pred) * 100 # 75.7
confusion_matrix(y_test, y_pred) 
pd.crosstab(y_test, y_pred)

