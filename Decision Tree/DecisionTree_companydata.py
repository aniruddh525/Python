# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 23:51:16 2021

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
comp_data=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Decision Tree\\Company_Data.csv")
comp_data.columns
comp_data.info()
comp_data.head()

# replacing numerical output variable to categorical, low , med and high

comp_data['Sales']= comp_data.apply(lambda row: 'low' if row.Sales<5 else ('med' if row.Sales>=5 and row.Sales < 10 else 'high' ) , axis=1)

# put label for categorical output variable

label_encoder = preprocessing.LabelEncoder()
comp_data['Sales']= label_encoder.fit_transform(comp_data['Sales']) 


# Getting dummy columns for categorical data
one_hot_data = pd.get_dummies(comp_data.iloc[:,[6,9,10]])
one_hot_data.columns

final_comp_data=pd.concat([comp_data.iloc[:,[0,1,2,3,4,5,7,8]],one_hot_data],axis=1)
final_comp_data.info()

#separateing IV and DV

x=final_comp_data.iloc[:,1:]
y=final_comp_data['Sales']

final_comp_data.columns
final_comp_data['Sales'].unique()
final_comp_data.Sales.value_counts()


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
fn=['CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'Age', 'Education', 'ShelveLoc_Bad', 'ShelveLoc_Good',
       'ShelveLoc_Medium', 'Urban_No', 'Urban_Yes', 'US_No', 'US_Yes']
cn=['med','high','low']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,feature_names = fn,class_names=cn,filled = True);

#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 
y_test.value_counts()

# getting the 2 way table to understand the correct and wrong predictions
pd.crosstab(y_test,preds) 

# Accuracy 
np.mean(preds==y_test) # 0.60


############### CART #########################

# Building Decision Tree Classifier using gini Criteria
model = DecisionTreeClassifier(criterion = 'gini',max_depth=4)
model.fit(x_train,y_train)

#PLot the decision tree
tree.plot_tree(model);

#PLot the decision tree with proper labels to give better understanding
fn=['CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'Age', 'Education', 'ShelveLoc_Bad', 'ShelveLoc_Good',
       'ShelveLoc_Medium', 'Urban_No', 'Urban_Yes', 'US_No', 'US_Yes']
cn=['med','high','low']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,feature_names = fn,class_names=cn,filled = True);

#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 
y_test.value_counts()

# getting the 2 way table to understand the correct and wrong predictions
pd.crosstab(y_test,preds) 

# Accuracy 
np.mean(preds==y_test) ## 0.56

