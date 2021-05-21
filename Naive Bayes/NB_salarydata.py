# -*- coding: utf-8 -*-
"""
Created on Fri May 21 21:16:08 2021

@author: anirudh.kumar.verma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.preprocessing import LabelEncoder


# load data
salary_train=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Naive Bayes\\new\\SalaryData_Train.csv")
salary_test=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Naive Bayes\\new\\SalaryData_Test.csv")

salary_train.columns
salary_train.info()
salary_train.head()
salary_train.describe()
salary_train.shape


X_cols=['workclass', 'education', 'maritalstatus',
       'occupation', 'relationship', 'race', 'sex', 'native']

le = LabelEncoder()
for i in X_cols:
    salary_train[i] = le.fit_transform(salary_train[i])
    salary_test[i] = le.fit_transform(salary_test[i])

X_train=salary_train[salary_train.columns[0:13]]
X_test=salary_test[salary_test.columns[0:13]]
Y_train=salary_train[salary_train.columns[13]]
Y_test=salary_test[salary_test.columns[13]]


#model building
NB_model = GaussianNB()
NB_model.fit(X_train, Y_train)

# predict
Y_pred=NB_model.predict(X_test)

# evaluate accuracy
np.mean(Y_test==Y_pred) # 0.79468
confusion_matrix(Y_test,Y_pred)
pd.crosstab(Y_test, Y_pred)
accuracy_score(Y_test, Y_pred) # 0.79468
