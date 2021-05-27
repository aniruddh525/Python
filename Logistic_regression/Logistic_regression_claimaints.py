# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:46:42 2021

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
claimants=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\claimants.csv")

# dropping the case number columns as it is not required
claimants.drop(["CASENUM"],inplace=True,axis = 1)
claimants.shape

# check for duplicated data
claimants.duplicated().sum()

# check for missing values
claimants.isna().sum()
claimants.info()

# Removing missing values 
claimants1 = claimants.dropna()
claimants1.shape

# study behaviour of data (EDA)
data_desc=claimants1.describe() # check for min and max value to cover max range.


# Data Distribution - Boxplot of continuous variables wrt to each category of categorical columns

sns.boxplot(y="LOSS",data=claimants1)
sns.boxplot(x="CLMAGE",data=claimants1)

sns.boxplot(x="ATTORNEY",y="CLMAGE",data=claimants1,palette="hls")
sns.boxplot(x="ATTORNEY",y="LOSS",data=claimants1,palette="hls")
sns.boxplot(x="CLMSEX",y="CLMAGE",data=claimants1,palette="hls")
sns.boxplot(x="CLMSEX",y="LOSS",data=claimants1,palette="hls")
sns.boxplot(x="SEATBELT",y="CLMAGE",data=claimants1,palette="hls")
sns.boxplot(x="SEATBELT",y="LOSS",data=claimants1,palette="hls")
sns.boxplot(x="CLMINSUR",y="CLMAGE",data=claimants1,palette="hls")
sns.boxplot(x="CLMINSUR",y="LOSS",data=claimants1,palette="hls")

sns.set_style(style='darkgrid') # optional - to change background
sns.pairplot(claimants1)
claimants1.corr() # correlation

# Dividing our data into input and output variables 
X = claimants1.iloc[:,1:]
Y = claimants1.iloc[:,0]


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
accuracy

#Classification report
print(classification_report(Y,y_pred))

TN=confusion_matrix[0,0]
TP=confusion_matrix[1,1]
FP=confusion_matrix[0,1]
FN=confusion_matrix[1,0]
Sensitivity=TP/(TP+FN)
Specificity=TN/(TN+FP)
precision=TP/(TP+FP)

# ROC Curve

fpr, tpr, thresholds = roc_curve(Y,model.predict_proba (X)[:,1])
roc_df=pd.DataFrame({'fpr':fpr,'tpr':tpr,'cutoff':thresholds})
roc_df
roc_grt_76=roc_df[roc_df['tpr']>=0.76]
auc = roc_auc_score(Y, y_pred)
auc
plt.plot(fpr, tpr, color='red', label='logit model ( area  = %0.2f)'%auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
plt.ylabel('True Positive Rate')


# splitting data into train and test
claimants_train,claimants_test=train_test_split(claimants1,test_size=0.2)
train_X=claimants_train.iloc[:,1:]
train_Y=claimants_train.iloc[:,0]
test_X=claimants_test.iloc[:,1:]
test_Y=claimants_test.iloc[:,0]

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

#Predict for testing dataset
y_test_pred = model_train.predict(test_X)
y_test_pred_prob = model_train.predict_proba(test_X)

y_pred_df_test= pd.DataFrame({'actual_test': test_Y,'predicted_test': y_test_pred})

pd.crosstab(y_pred_df_test.actual_test, y_pred_df_test.predicted_test)
cm_test=confusion_matrix(test_Y,y_test_pred)
cm_test

accuracy_test=(cm_test[0,0]+cm_test[1,1])/cm_test.sum()



######### another way of implementing Log. Reg is through stasmodels pkg ######

# first perform imputations

claimants2=claimants.copy()
claimants2.isnull().sum()

# filling the missing value with most occuring value 
claimants2.iloc[:,0:4].columns   
claimants2.iloc[:,0:4] = claimants2.iloc[:,0:4].apply(lambda x:x.fillna(x.value_counts().index[0]))
claimants2.isnull().sum()

# filling the missing value with mean of that column
claimants2.iloc[:,4:] = claimants2.iloc[:,4:].apply(lambda x:x.fillna(x.mean()))

claimants2.isnull().sum()


logit_model = smf.logit('ATTORNEY~CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT',data = claimants2).fit()
logit_model.summary()

y_pred_sm=logit_model.predict(claimants2)

# creating new column for predicted variable
claimants2['Pred_value']=y_pred_sm
claimants2["att_pred"]=0

# apply cuttof value and get final predcited values in 0 and 1
claimants2.loc[claimants2.Pred_value>=0.5,"att_pred"] = 1


cm_sm = confusion_matrix(claimants2.ATTORNEY,claimants2.att_pred)
cm_sm

# accuracy
accuracy_sm=(cm_sm[0,0]+cm_sm[1,1])/cm_sm.sum()
accuracy_sm








