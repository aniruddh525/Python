# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 12:54:58 2021

@author: anirudh.kumar.verma
"""


import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import math as mt
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# load data
bank=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Logistic regression\\bank-full.csv")
bank.columns

# study behaviour of data (EDA)
bank.info()
bank.job.value_counts() # 12 disticnt 
bank.marital.value_counts() # 3
bank.education.value_counts() # 4
bank.default.value_counts() # 2
bank.housing.value_counts() # 2
bank.loan.value_counts() # 2
bank.contact.value_counts() # 3
bank.month.value_counts() # 12
bank.poutcome.value_counts() # 4
bank.y.value_counts() # 2

data_desc=bank.describe()

sns.boxplot(y="age",data=bank)
sns.boxplot(x="balance",data=bank)

# check for duplicated data
bank.duplicated().sum()

# check for missing values
bank.isna().sum()
bank.info()

# Data Distribution - Boxplot 

sns.boxplot(y="age",data=bank)
sns.boxplot(x="balance",data=bank)

# create dummy variable for categorical data

dummy_2=pd.get_dummies(bank[["default","housing",'loan']],drop_first=True)
dummy_3=pd.get_dummies(bank[["job","marital",'education','contact','month','poutcome']])


# Dropping the columns for which we have created dummies
bank.drop(["default","housing","loan","job","marital","education","contact","month",'poutcome'],inplace=True,axis = 1)

# adding the columns to the salary data frame 
bank1 = pd.concat([dummy_2,dummy_3,bank],axis=1)

bank1.loc[bank1.y=="yes","y"] = 1
bank1.loc[bank1.y=="no","y"] = 0

bank1['y'] = bank1['y'].astype(str).astype(int)
bank1.info()


# Dividing our data into input and output variables 
X = bank1.iloc[:,0:48]
Y = bank1.iloc[:,48]

# build model
model = LogisticRegression()
model.fit(X,Y)


#Predict for X dataset
y_pred = model.predict(X)
y_pred_prob = model.predict_proba(X)

y_pred_df= pd.DataFrame({'actual': Y,'predicted': model.predict(X)})


# Confusion Matrix 
confusion_matrix = confusion_matrix(Y,y_pred)
confusion_matrix

# another way to create confusion matrix
pd.crosstab(y_pred_df.actual,y_pred_df.predicted)

# accuracy
accuracy=(confusion_matrix[0,0]+confusion_matrix[1,1])/confusion_matrix.sum()
accuracy # 0.891

#Classification report
print(classification_report(Y,y_pred))

TN=confusion_matrix[0,0]
TP=confusion_matrix[1,1]
FP=confusion_matrix[0,1]
FN=confusion_matrix[1,0]
Sensitivity=TP/(TP+FN) # 0.22
Specificity=TN/(TN+FP) # 0.98
precision=TP/(TP+FP) # 0.60

# ROC Curve

fpr, tpr, thresholds = roc_curve(Y,model.predict_proba (X)[:,1])
roc_df=pd.DataFrame({'fpr':fpr,'tpr':tpr,'cutoff':thresholds})
roc_df
roc_grt_76=roc_df[roc_df['tpr']>=0.76]
auc = roc_auc_score(Y, y_pred)
auc #0.60

plt.plot(fpr, tpr, color='red', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')


# splitting data into train and test

bank_train,bank_test=train_test_split(bank1,test_size=0.2)
train_X=bank_train.iloc[:,0:48]
train_Y=bank_train.iloc[:,48]
test_X=bank_test.iloc[:,0:48]
test_Y=bank_test.iloc[:,48]

#build model on training dataset
model_train = LogisticRegression()
model_train.fit(train_X,train_Y)

#Predict for training dataset
y_train_pred = model_train.predict(train_X)
y_train_pred_prob = model_train.predict_proba(train_X)

y_pred_df_train= pd.DataFrame({'actual_train': train_Y,'predicted_train': y_train_pred})

pd.crosstab(y_pred_df_train.actual_train, y_pred_df_train.predicted_train)
cm_train=confusion_matrix(train_Y,y_train_pred)
cm_train

accuracy_train=(cm_train[0,0]+cm_train[1,1])/cm_train.sum()
accuracy_train # 0.892

#Predict for testing dataset
y_test_pred = model_train.predict(test_X)
y_test_pred_prob = model_train.predict_proba(test_X)

y_pred_df_test= pd.DataFrame({'actual_test': test_Y,'predicted_test': y_test_pred})

pd.crosstab(y_pred_df_test.actual_test, y_pred_df_test.predicted_test)
cm_test=confusion_matrix(test_Y,y_test_pred)
cm_test

accuracy_test=(cm_test[0,0]+cm_test[1,1])/cm_test.sum()
accuracy_test # 0.885
