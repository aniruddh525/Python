# -*- coding: utf-8 -*-
"""
Created on Wed May 12 12:53:20 2021

@author: anirudh.kumar.verma
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import tree

# load data
company=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Decision Tree\\Company_Data.csv")
company.columns
company.head()
company.info()
company.describe()


# replacing numerical output variable to categorical, low , med and high

company['Sales']= company.apply(lambda row: 'low' if row.Sales<5 else ('med' if row.Sales>=5 and row.Sales < 10 else 'high' ) , axis=1)

# put label for categorical output variable

label_encoder = preprocessing.LabelEncoder()
company['Sales']= label_encoder.fit_transform(company['Sales']) 


# Getting dummy columns for categorical data
one_hot_data = pd.get_dummies(company.iloc[:,[6,9,10]])
one_hot_data.columns

final_company=pd.concat([company.iloc[:,[0,1,2,3,4,5,7,8]],one_hot_data],axis=1)
final_company.info()

#separateing IV and DV

x=final_company.iloc[:,1:]
y=final_company['Sales']

final_company.columns
final_company['Sales'].unique()
final_company.Sales.value_counts()


# Splitting data into training and testing data set, 
#random_state to keep same train/test data everyrun (equv to set.seed)
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)

num_trees = 150
RF_model = RandomForestClassifier(n_estimators=num_trees)
RF_model.fit(x_train,y_train)


#Predicting on test data
preds = RF_model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 
y_test.value_counts()

# getting the 2 way table to understand the correct and wrong predictions
pd.crosstab(y_test,preds) 

# Accuracy 
np.mean(preds==y_test) # 0.5875

# accuracy using cross validations

kfold = KFold(n_splits=10, random_state=7,shuffle=True)
results = cross_val_score(RF_model, x, y, cv=kfold)
results.mean() # 0.6975



