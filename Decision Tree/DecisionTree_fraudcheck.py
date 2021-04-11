# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 11:27:24 2021

@author: anirudh.kumar.verma
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing


# load data
fraud=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Decision Tree\\Fraud_check.csv")
fraud.columns
fraud.info()
fraud.head()

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

x=final_fraud.iloc[:,[0,1,2,3,4,5,6,8,9]]
y=final_fraud['Taxable_Income_Good']

final_fraud.columns
final_fraud['Taxable_Income_Good'].unique()
final_fraud.Taxable_Income_Good.value_counts()


# Splitting data into training and testing data set, 
#random_state to keep same train/test data everyrun (equv to set.seed)
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


############### C5.0 #########################

# Building Decision Tree Classifier using Entropy Criteria
model = DecisionTreeClassifier(criterion = 'entropy',max_depth=4)
model.fit(x_train,y_train)

#PLot the decision tree
tree.plot_tree(model);

#PLot the decision tree with proper labels to give better understanding
fn=['City_Population', 'Work_Experience', 'Undergrad_NO', 'Undergrad_YES',
       'Marital_Status_Divorced', 'Marital_Status_Married',
       'Marital_Status_Single', 'Urban_NO',
       'Urban_YES']
cn=['Risky', 'Good']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,feature_names = fn,class_names=cn,filled = True);

#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 
y_test.value_counts()

# getting the 2 way table to understand the correct and wrong predictions
pd.crosstab(y_test,preds) 

# Accuracy 
np.mean(preds==y_test) # 0.76666


############### CART #########################

# Building Decision Tree Classifier using gini Criteria
model = DecisionTreeClassifier(criterion = 'gini',max_depth=4)
model.fit(x_train,y_train)

#PLot the decision tree
tree.plot_tree(model);

#PLot the decision tree with proper labels to give better understanding
fn=['City_Population', 'Work_Experience', 'Undergrad_NO', 'Undergrad_YES',
       'Marital_Status_Divorced', 'Marital_Status_Married',
       'Marital_Status_Single', 'Urban_NO',
       'Urban_YES']
cn=['Risky', 'Good']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,feature_names = fn,class_names=cn,filled = True);

#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 
y_test.value_counts()

# getting the 2 way table to understand the correct and wrong predictions
pd.crosstab(y_test,preds) 

# Accuracy 
np.mean(preds==y_test) ## 0.75833

