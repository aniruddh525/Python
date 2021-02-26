# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 22:49:15 2021

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
startup=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Multiple Linear Regression\\50_Startups.csv")

# check for duplicated data
startup.duplicated().sum() # 0 - no duplicate

# convert categorical variable to numeric variable 
startup=pd.get_dummies(data=startup,drop_first=True)

# rename columns
startup.columns
startup.rename(columns={'R&D Spend': 'RnD','Marketing Spend':'Market',
                        'State_Florida':'St_F','State_New York':'St_N'}, inplace=True)

startup.head()

# check for missing values
startup.isna().sum()
startup.info()

# study behaviour of data (EDA)
data_desc=startup.describe() # check for min and max value to cover max range.
# scatter plot
sns.set_style(style='darkgrid') # optional - to change background
sns.pairplot(startup)
data_corr=startup.corr() # correlation

# build model
startup.columns
model=smf.ols('Profit~RnD+Administration+Market+St_F+St_N',data=startup).fit()
# chcek parameters
model.summary()
model.params    
model.pvalues # admin , market , st_f , st_n - all insig
model.rsquared # 0.95
model.rsquared_adj
model.aic # 1062.76
model.resid 

# chcek for insignificant variable if independtly they are significant

model_ad = smf.ols('Profit~Administration',data=startup).fit()
model_ad.summary() # insig
model_mar = smf.ols('Profit~Market',data=startup).fit()
model_mar.summary() # sig
model_mar_ad=smf.ols('Profit~Administration+Market',data=startup).fit()
model_mar_ad.summary() # both sig
# doesnt look colinear , may be some other reason for being insignificant

# check collinearity by calculating VIF for each IV in R Vif(model)

rsq_rnd = smf.ols('RnD~Administration+Market+St_F+St_N',data=startup).fit().rsquared  
vif_rnd = 1/(1-rsq_rnd) 

rsq_adm = smf.ols('Administration~RnD+Market+St_F+St_N',data=startup).fit().rsquared  
vif_adm = 1/(1-rsq_adm) 

rsq_mar = smf.ols('Market~RnD+Administration+St_F+St_N',data=startup).fit().rsquared  
vif_mar = 1/(1-rsq_mar)

rsq_stf = smf.ols('St_F~RnD+Administration+Market+St_N',data=startup).fit().rsquared  
vif_stf = 1/(1-rsq_stf)

rsq_stn = smf.ols('St_N~RnD+Administration+Market+St_F',data=startup).fit().rsquared  
vif_stn = 1/(1-rsq_stn)

# Storing vif values in a data frame
d1 = {'Variables':['RnD','Admin','mar','st_f','st_N'],
      'VIF':[vif_rnd,vif_adm,vif_mar,vif_stf,vif_stn]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame # no colinearity as all are very low values.


###### Model Validation ############

model.resid.mean() # zero
model.resid.hist() # not very normal
qqplot=sm.qqplot(model.resid,line='q') # just ok , have outlier
sns.scatterplot(startup.Profit,model.fittedvalues) # almost linear
sns.scatterplot(model.fittedvalues,model.resid) # no pattern - ok
# Residual Vs Regressors 
sm.graphics.plot_regress_exog(model,'RnD') # pat
sm.graphics.plot_regress_exog(model,'Administration')
sm.graphics.plot_regress_exog(model,'Market') # slight pat
sm.graphics.plot_regress_exog(model,'St_F') 
sm.graphics.plot_regress_exog(model,'St_N')


##### Deletion diagnostic #######

# Cook's distance
(c, _)=model.get_influence().cooks_distance
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(startup)), np.round(c, 3)) 
(np.argmax(c),np.max(c)) #(49, 0.2639594358718258) - no outlier
# High influence point
influence_plot(model) # 49
k = startup.shape[1]
n = startup.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff

# chcek influencer/outlier data
startup[startup.index.isin([49])] 
startup.head(10)

########## improve model #########

# removing categorical variable as its highly insiginificant and not
# adding any value

###### Iteration 1 ############

model1=smf.ols('Profit~RnD+Administration+Market',data=startup).fit()
# chcek parameters
model1.summary() 
model1.pvalues # doors not good
model1.rsquared
model1.rsquared_adj
model1.aic

#### perform model validation #####

model1.resid.mean() #  zero
model1.resid.hist() # not fully OK
qqplot=sm.qqplot(model1.resid,line='q') # better than previous
sns.scatterplot(startup.Profit,model1.fittedvalues) # almost normal 
sns.scatterplot(model1.fittedvalues,model1.resid) # OK
# Residual Vs Regressors 
sm.graphics.plot_regress_exog(model1,'RnD') # pat
sm.graphics.plot_regress_exog(model1,'Administration')
sm.graphics.plot_regress_exog(model1,'Market') # slight pat


# chcek for other influencers
(c, _)=model1.get_influence().cooks_distance
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(startup)), np.round(c, 3)) 
(np.argmax(c),np.max(c)) # no outlier

influence_plot(model1) # no influenter
k = startup.shape[1]
n = startup.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff

# Variable transformation tried with
# square, square root and log for 'admininstartion' and 'marketing spend' and diff combination
# but nothing worked out so the best option would be to remove variables

model2=smf.ols('Profit~RnD+Market',data=startup).fit() # market nearly significant, best Rsq and AIC value
model3=smf.ols('Profit~RnD',data=startup).fit()

# chcek parameters
model2.summary() 
model2.pvalues 
model2.rsquared # 0.95
model2.rsquared_adj
model3.aic


# ALL OK

# chcek model performance with RMSE and MAPE value

# splitting data into train and test
startup_train,startup_test=train_test_split(startup,test_size=0.2) # 20% testing data

# prepraing model on training data
model_train=smf.ols('Profit~RnD',data=startup_train).fit()
model_train.summary()

# get RMSE (root mean square error) value for training data set
RMSE_train=np.sqrt(np.mean(model_train.resid*model_train.resid))
RMSE_train # 9111.049

def MAPE(pred,actual):
    return np.mean(abs(pred-actual)/actual)*100

MAPE(model_train.fittedvalues,startup_train.Profit) # 11.8762

#Predicting on test data
test_pred=model_train.predict(startup_test) # fitted values on test data
test_resid=startup_test.Profit-test_pred # errors on test data

RMSE_test=np.sqrt(np.mean(test_resid*test_resid))
RMSE_test # 10055.14
MAPE(test_pred,startup_test.Profit) # 7.79095


