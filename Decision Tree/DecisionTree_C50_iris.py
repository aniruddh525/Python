# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 23:59:01 2021

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
iris=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\iris.csv")
iris.columns
iris.info()
iris.head()

# put label for categorical output variable

label_encoder = preprocessing.LabelEncoder()
iris['Species']= label_encoder.fit_transform(iris['Species']) 

x=iris.iloc[:,0:4]
y=iris['Species']

iris['Species'].unique()

iris.Species.value_counts()

colnames = list(iris.columns)
colnames

# Splitting data into training and testing data set, 
#random_state to keep same train/test data everyrun (equv to set.seed)
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)


# Building Decision Tree Classifier using Entropy Criteria
model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)

#PLot the decision tree
tree.plot_tree(model);

#PLot the decision tree with proper labels to give better understanding
fn=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)']
cn=['setosa', 'versicolor', 'virginica']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,feature_names = fn,class_names=cn,filled = True);

#Predicting on test data
preds = model.predict(x_test) # predicting on test data set 
pd.Series(preds).value_counts() # getting the count of each category 

# getting the 2 way table to understand the correct and wrong predictions
pd.crosstab(y_test,preds) 

# Accuracy 
np.mean(preds==y_test)

