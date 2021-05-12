# -*- coding: utf-8 -*-
"""
Created on Wed May 12 11:16:21 2021

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
fraud=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Decision Tree\\Fraud_check.csv")
fraud.columns
fraud.head()
fraud.info()
fraud.describe()


#rename columns

fraud.rename(columns={"Marital.Status": "Marital_Status", "Taxable.Income": "Taxable_Income","City.Population":"City_Population","Work.Experience":"Work_Experience"},inplace=True)

# replacing numerical output variable to categorical, Risky  and good

fraud['Taxable_Income']= fraud.apply(lambda row: 'Risky' if row.Taxable_Income<=30000 else 'Good' , axis=1)

# Getting dummy columns for categorical data
one_hot_data = pd.get_dummies(fraud)
one_hot_data.columns
final_fraud=one_hot_data.drop("Taxable_Income_Risky",axis=1)
final_fraud.info()

#separateing IV and DV

X=final_fraud.iloc[:,[0,1,2,3,4,5,6,8,9]]
Y=final_fraud['Taxable_Income_Good']

final_fraud.columns
final_fraud['Taxable_Income_Good'].unique()
final_fraud.Taxable_Income_Good.value_counts()

x=np.array(X)
y=np.array(Y)


# Splitting data into training and testing data set, 
#random_state to keep same train/test data everyrun (equv to set.seed)
x_train, x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2,random_state=40)




num_trees = 150
max_features = 3
RF_model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
RF_model.fit(x_train,y_train)



#Predicting on test data
preds = RF_model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 
y_test.value_counts()

# getting the 2 way table to understand the correct and wrong predictions
pd.crosstab(y_test,preds) 

# Accuracy 
np.mean(preds==y_test) # 0.7083

# accuracy using cross validations

kfold = KFold(n_splits=10, random_state=7,shuffle=True)
results = cross_val_score(RF_model, x, y, cv=kfold)
results.mean() # 0.73
