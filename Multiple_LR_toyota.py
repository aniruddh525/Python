# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:30:43 2021

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
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.model_selection import train_test_split 

# load data
toyota=pd.read_excel("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\Toyota.xlsx")

# select only needed columns
col_mapping = [f"{c[0]}:{c[1]}" for c in enumerate(toyota.columns)]
toyota=toyota.iloc[:,[2,3,6,8,12,13,15,16,17]]
toyota.rename(columns={'Age_08_04': 'Age'}, inplace=True)
toyota.head()

# check for missing values
toyota.isna().sum()
toyota.info()
# get index of missing values
toyota[toyota['HP'].isnull()].index
# drop null rows
toyota=toyota.drop([7,38]).reset_index()
toyota=toyota.drop(['index'],axis=1)

# study behaviour of data (EDA)
data_desc=toyota.describe() # check for min and max value to cover max range.
# scatter plot
sns.set_style(style='darkgrid') # optional - to change background
sns.pairplot(toyota)
data_corr=toyota.corr() # correlation

# build model
toyota.columns
model=smf.ols('Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota).fit()
# chcek parameters
model.summary()
model.params    
model.pvalues
model.rsquared
model.rsquared_adj
model.aic
model.resid # residuals/errors

# chcek for insignificant variable if independtly they are significant

model_cc = smf.ols('Price~cc',data=toyota).fit()
model_cc.summary()
model_doors = smf.ols('Price~Doors',data=toyota).fit()
model_doors.summary()
model_cc_doors=smf.ols('Price~cc+Doors',data=toyota).fit()
model_cc_doors.summary()
# doesnt look colinear , may be some other reason for being insignificant

###### Model Validation ############


model.resid.mean() # mean of residuals should be zero
model.resid.hist() # normality of errors
qqplot=sm.qqplot(model.resid,line='q') # residual q-q plot
sns.scatterplot(toyota.Price,model.fittedvalues) # observed v fitted
sns.scatterplot(model.fittedvalues,model.resid) # fitted value v residuals
# Residual Vs Regressors 
fig = plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'Age',fig=fig) # not good
sm.graphics.plot_regress_exog(model,'KM')
sm.graphics.plot_regress_exog(model,'HP')
sm.graphics.plot_regress_exog(model,'cc')
sm.graphics.plot_regress_exog(model,'Doors')
sm.graphics.plot_regress_exog(model,'Gears')
sm.graphics.plot_regress_exog(model,'Quarterly_Tax')
sm.graphics.plot_regress_exog(model,'Weight')

##### Deletion diagnostic #######

# Cook's distance
(c, _)=model.get_influence().cooks_distance
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(toyota)), np.round(c, 3)) 
(np.argmax(c),np.max(c))
# High influence point
influence_plot(model)
k = toyota.shape[1]
n = toyota.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff

# chcek influencer/outlier data
toyota[toyota.index.isin([78])] 
toyota.head(10)

########## improve model #########

toyota1=toyota.drop([78]).reset_index()
toyota1=toyota1.drop(['index'],axis=1)

###### Iteration 1 ############

#### build model again on new data #######

model1=smf.ols('Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota1).fit()
# chcek parameters
model1.summary() 
model1.pvalues # doors not good
model1.rsquared
model1.rsquared_adj
model1.aic

#### perform model validation #####

model1.resid.mean() # mean of residuals should be zero
model1.resid.hist() # normality of errors
qqplot=sm.qqplot(model1.resid,line='q') # residual q-q plot
sns.scatterplot(toyota1.Price,model1.fittedvalues) # observed v fitted
sns.scatterplot(model1.fittedvalues,model1.resid) # fitted value v residuals - not good
# Residual Vs Regressors 
sm.graphics.plot_regress_exog(model1,'Age') # not good
sm.graphics.plot_regress_exog(model1,'KM')
sm.graphics.plot_regress_exog(model1,'HP')
sm.graphics.plot_regress_exog(model1,'cc') # good now
sm.graphics.plot_regress_exog(model1,'Doors')
sm.graphics.plot_regress_exog(model1,'Gears')
sm.graphics.plot_regress_exog(model1,'Quarterly_Tax')
sm.graphics.plot_regress_exog(model1,'Weight')

# chcek for other influencers
(c, _)=model1.get_influence().cooks_distance
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(toyota1)), np.round(c, 3)) 
(np.argmax(c),np.max(c)) # 218 , 957

influence_plot(model1) # 218, 957
k = toyota1.shape[1]
n = toyota1.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff

toyota1[toyota1.index.isin([218,957])] 
toyota1.head(10)

toyota2=toyota1.drop([218,957]).reset_index()
toyota2=toyota2.drop(['index'],axis=1)

###### Iteration 2 ############

model2=smf.ols('Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=toyota2).fit()
model2.summary() 
model2.pvalues # doors ok now
model2.rsquared
model2.rsquared_adj
model1.aic

# Model validation

model2.resid.mean() # mean of residuals should be zero
model2.resid.hist() # normality of errors
qqplot=sm.qqplot(model2.resid,line='q') # residual q-q plot
sns.scatterplot(toyota2.Price,model2.fittedvalues) # observed v fitted
sns.scatterplot(model2.fittedvalues,model2.resid) # fitted value v residuals - not good
# Residual Vs Regressors 
sm.graphics.plot_regress_exog(model2,'Age') # still not good
sm.graphics.plot_regress_exog(model2,'KM')
sm.graphics.plot_regress_exog(model2,'HP')
sm.graphics.plot_regress_exog(model2,'cc') 
sm.graphics.plot_regress_exog(model2,'Doors')
sm.graphics.plot_regress_exog(model2,'Gears')
sm.graphics.plot_regress_exog(model2,'Quarterly_Tax')
sm.graphics.plot_regress_exog(model2,'Weight')

# chcek for other influencers
(c, _)=model2.get_influence().cooks_distance
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(toyota2)), np.round(c, 3)) 
(np.argmax(c),np.max(c)) # (597, 0.31685843275493203) - no action now

influence_plot(model2) # 597
k = toyota2.shape[1]
n = toyota2.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff

toyota2[toyota2.index.isin([597])] 
toyota2.head(10) ### leave it for now


# Variable transformation

toyota2['Age2']=toyota2.Age*toyota2.Age


###### Iteration 3 ############

model3=smf.ols('Price~Age+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight+Age2',data=toyota2).fit()
model3.summary() 
model3.pvalues # doors again not good
model3.rsquared
model3.rsquared_adj
model3.aic

# Model validation

model3.resid.mean() # mean of residuals should be zero
model3.resid.hist() # normality of errors
qqplot=sm.qqplot(model3.resid,line='q') # residual q-q plot
sns.scatterplot(toyota2.Price,model3.fittedvalues) # observed v fitted
sns.scatterplot(model3.fittedvalues,model3.resid) # fitted value v residuals - now good
# Residual Vs Regressors 
sm.graphics.plot_regress_exog(model3,'Age') # now good
sm.graphics.plot_regress_exog(model3,'KM')
sm.graphics.plot_regress_exog(model3,'HP')
sm.graphics.plot_regress_exog(model3,'cc') 
sm.graphics.plot_regress_exog(model3,'Doors')
sm.graphics.plot_regress_exog(model3,'Gears')
sm.graphics.plot_regress_exog(model3,'Quarterly_Tax')
sm.graphics.plot_regress_exog(model3,'Weight')
sm.graphics.plot_regress_exog(model3,'Age2')

# chcek for other influencers
(c, _)=model3.get_influence().cooks_distance
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(toyota2)), np.round(c, 3)) 
(np.argmax(c),np.max(c)) # no new outlier

influence_plot(model3) # no new influencer
k = toyota2.shape[1]
n = toyota2.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff



# removing 'doors' as its insignificant (constant value) and building final model

###### Iteration 4 ############

modelf=smf.ols('Price~Age+KM+HP+cc+Gears+Quarterly_Tax+Weight+Age2',data=toyota2).fit()
modelf.summary() 
modelf.pvalues 
modelf.rsquared
modelf.rsquared_adj
model3.aic

# Model validation

modelf.resid.mean() # mean of residuals should be zero
modelf.resid.hist() # normality of errors
qqplot=sm.qqplot(modelf.resid,line='q') # residual q-q plot
sns.scatterplot(toyota2.Price,modelf.fittedvalues) # observed v fitted
sns.scatterplot(modelf.fittedvalues,modelf.resid) # fitted value v residuals - now good
# Residual Vs Regressors 
sm.graphics.plot_regress_exog(modelf,'Age') # now good
sm.graphics.plot_regress_exog(modelf,'KM')
sm.graphics.plot_regress_exog(modelf,'HP')
sm.graphics.plot_regress_exog(modelf,'cc') 
# sm.graphics.plot_regress_exog(modelf,'Doors')
sm.graphics.plot_regress_exog(modelf,'Gears')
sm.graphics.plot_regress_exog(modelf,'Quarterly_Tax')
sm.graphics.plot_regress_exog(modelf,'Weight')
sm.graphics.plot_regress_exog(modelf,'Age2')

# chcek for other influencers
(c, _)=modelf.get_influence().cooks_distance
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(toyota2)), np.round(c, 3)) 
(np.argmax(c),np.max(c)) # no new outlier

influence_plot(modelf) # no new influencer
k = toyota2.shape[1]
n = toyota2.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff

# ALL OK

# chcek model performance with RMSE and MAPE value

# splitting data into train and test
toyota_train,toyota_test=train_test_split(toyota2,test_size=0.2) # 20% testing data

# prepraing model on training data
model_train=smf.ols('Price~Age+KM+HP+cc+Gears+Quarterly_Tax+Weight+Age2',data=toyota_train).fit()
model_train.summary()

# get RMSE (root mean square error) value for training data set
RMSE_train=np.sqrt(np.mean(model_train.resid*model_train.resid))
RMSE_train

def MAPE(pred,actual):
    return np.mean(abs(pred-actual)/actual)*100

MAPE(model_train.fittedvalues,toyota_train.Price)

#Predicting on test data
test_pred=model_train.predict(toyota_test) # fitted values on test data
test_resid=toyota_test.Price-test_pred # errors on test data

RMSE_test=np.sqrt(np.mean(test_resid*test_resid))
RMSE_test
MAPE(test_pred,toyota_test.Price)

