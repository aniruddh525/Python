# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 10:30:08 2021

@author: anirudh.kumar.verma
"""


import pandas as pd
import numpy as np
import scipy 
from scipy import stats
import statsmodels.stats.descriptivestats as sd
import seaborn as sns
import math as mt
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.model_selection import train_test_split 

######## cars MPG problem ####

# load data
cars=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\Cars.csv")
cars.head()

# chcek for missing values
cars.info()
cars.isna().sum()

# study behaviour of data (EDA)

cars.describe() # check for min and max value to cover max range.
# scatter plot
sns.set_style(style='darkgrid') # optional - to change background
sns.pairplot(cars)
cars.corr() # correlation

# build model
model=smf.ols('MPG~HP+VOL+SP+WT',data=cars).fit()
# chcek parameters
model.summary()
model.params    
model.pvalues
model.rsquared
model.rsquared_adj
model.aic
model.resid # residuals/errors

# chcek for insignificant variable if independtly they are significant

model_Vol = smf.ols('MPG~VOL',data=cars).fit()
model_Vol.summary()
model_Wt = smf.ols('MPG~WT',data=cars).fit()
model_Wt.summary()
model_wt_vol=smf.ols('MPG~VOL+WT',data=cars).fit()
model_wt_vol.summary()

# check collinearity by calculating VIF for each IV in R Vif(model)

rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared  
vif_hp = 1/(1-rsq_hp) 

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame # WT and VOL are > 20 so they are colinear but no action now. just note

# another quick way to find Vif --- Not working

# from statsmodels.stats.outliers_influence import variance_inflation_factor
# cars_x=cars[['HP','SP','VOL','WT']]
# vif_new=[variance_inflation_factor(cars_x.values,i) for i in range(cars_x.shape[1])]
# pd.DataFrame({'vif_new':vif_new[0:]}, index=cars_x.columns).T


###### Model Validation ############

# mean of residuals should be zero
model.resid.mean()

# errors should follow normal dis
model.resid.hist()
# Q-Q plot -- plot b/w actual value and theoritcal quantiles (z value)
qqplot=sm.qqplot(model.resid,line='q') # line = 45 to draw the diagnoal line
plt.title("Normal Q-Q plot of residuals")

# observed v fitted - should give a linear line
sns.scatterplot(cars.MPG,model.fittedvalues)

# fitted value v residuals - no pattern , good model
sns.scatterplot(model.fittedvalues,model.resid)

# Residual Plot for Homoscedasticity (almost same as fitted value v residuals)
def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std() # standardize values (z score)

sns.scatterplot(x=get_standardized_values(model.fittedvalues),y=get_standardized_values(model.resid))

# Residual Vs Regressors - 4 plots , e v X (2nd plot) - no pattern , good model

fig = plt.figure(figsize=(15,8))
sm.graphics.plot_regress_exog(model,'VOL',fig=fig)
sm.graphics.plot_regress_exog(model,'HP')
sm.graphics.plot_regress_exog(model,'SP')
sm.graphics.plot_regress_exog(model,'WT')

# added varaible plot - same as third graph of previous plot - should not be parallel lines , means no effect
sm.graphics.plot_partregress_grid(model)

##### Deletion diagnostic #######

# Cook's distance

(c, _)=model.get_influence().cooks_distance

fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(cars)), np.round(c, 3)) 

#index and value of influencer where c is more than 1/.5
(np.argmax(c),np.max(c))

# High influence point
influence_plot(model)
# to find cut off value for influnecer in influence plot
k = cars.shape[1]
n = cars.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff

# chcek influencer/outlier points as why it is influencer
cars[cars.index.isin([76])] 
#See the differences in HP and other variable values
cars.head()

########## improve model #########

#Discard the data points which are influencers and reasign the row number (reset_index())
cars1=cars.drop(cars.index[[76]],axis=0).reset_index() # one more column 'index, gets added with old index no.
#Drop the original index
cars1=cars1.drop(['index'],axis=1)

###### Iteration 1 ############

#### build model again on new data #######

model1=smf.ols('MPG~HP+VOL+SP+WT',data=cars1).fit()
# chcek parameters
model1.summary() 
model1.pvalues # still Vol and Wt are insignificant
model1.rsquared
model1.rsquared_adj
model1.aic


#### perform model validation #####

# chcek for other influencers
# Cook's distance

(c, _)=model1.get_influence().cooks_distance

fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(cars)), np.round(c, 3)) 

#index and value of influencer where c is more than 1/.5
(np.argmax(c),np.max(c)) # value is less than 1 so stop this for now
# High influence point
influence_plot(model1) # 76 seems outlier
# to find cut off value for influnecer in influence plot
k = cars1.shape[1]
n = cars1.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff

# Variable transformation

HP2=cars1.HP*cars1.HP
SP2=cars1.SP*cars1.SP

cars2=cars1
cars2['HP2']=HP2
cars2['SP2']=SP2

# model building on new data

model2=smf.ols('MPG~HP+VOL+SP+WT+HP2+SP2',data=cars2).fit()
# chcek parameters
model2.summary() 
model2.pvalues # still Vol and Wt are insignificant
model2.rsquared
model2.rsquared_adj
model2.aic

## validation

model2.resid.mean() # residual mean
model2.resid.hist() # residual histogram
qqplot=sm.qqplot(model2.resid,line='q') # q-q plot for error 
sns.scatterplot(cars.MPG,model2.fittedvalues) # y v Yhat 
sns.scatterplot(model2.fittedvalues,model2.resid) # e v y
# residual v Xi

sm.graphics.plot_regress_exog(model2,'VOL')
sm.graphics.plot_regress_exog(model2,'HP')
sm.graphics.plot_regress_exog(model2,'SP')
sm.graphics.plot_regress_exog(model2,'WT')
sm.graphics.plot_regress_exog(model2,'HP2')
sm.graphics.plot_regress_exog(model2,'SP2')


# chcek for other influencers
(c, _)=model2.get_influence().cooks_distance
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(cars)), np.round(c, 3)) 
(np.argmax(c),np.max(c)) # 3 values more than 1
influence_plot(model2) #
k = cars2.shape[1]
n = cars2.shape[0]
leverage_cutoff = 3*((k + 1)/n)
leverage_cutoff

########## improve model #########
cars3=cars2.drop(cars2.index[[65,77,79]],axis=0).reset_index() # one more column 'index, gets added with old index no.
cars3=cars3.drop(['index'],axis=1)

# model building on new data

model3=smf.ols('MPG~HP+VOL+SP+WT+HP2+SP2',data=cars3).fit()
# chcek parameters
model3.summary() 
model3.pvalues 
model2.rsquared
model2.rsquared_adj
model2.aic

#### removing colinear variable ######
# which one to remove - VOL or WT - should be based on greater r suqre and lower aic

model3_wt=smf.ols('MPG~HP+SP+WT+HP2+SP2',data=cars3).fit()
model3_wt.summary()
model3_wt.pvalues  # all significant
model3_wt.rsquared
model3_wt.rsquared_adj
model3_wt.aic

model3_vol=smf.ols('MPG~HP+SP+VOL+HP2+SP2',data=cars3).fit()
model3_vol.summary()
model3_vol.pvalues 
model3_vol.rsquared
model3_vol.rsquared_adj
model3_vol.aic

# with Vol , r squ is more and aic is low so will go with vol.
# so final model

model_final=smf.ols('MPG~HP+SP+VOL+HP2+SP2',data=cars3).fit()
model_final.summary()

# chcek model performance with RMSE value

# splitting data into train and test
cars_train,cars_test=train_test_split(cars3,test_size=0.2) # 20% testing data

# prepraing model on training data
model_train=smf.ols('MPG~HP+SP+VOL+HP2+SP2',data=cars_train).fit()
model_train.summary()

# get RMSE (root mean square error) value for training data set
RMSE_train=np.sqrt(np.mean(model_train.resid*model_train.resid))
RMSE_train
# get MAPE(mean abs percentage error) value
def MAPE(pred,actual):
    return np.mean(abs(pred-actual)/actual)*100

MAPE(model_train.fittedvalues,cars_train.MPG)

#Predicting on test data

test_pred=model_train.predict(cars_test) # fitted values on test data
test_resid=cars_test.MPG-test_pred # errors on test data

RMSE_test=np.sqrt(np.mean(test_resid*test_resid))
RMSE_test

MAPE(test_pred,cars_test.MPG)

# predicting on new set of data

New_data=pd.DataFrame({'HP':40,'SP':102,'VOL':95,'WT':35,'HP2':1600,'SP2':10404},index=[1])
predicted_value=model_train.predict(New_data)
predicted_value

# but this predicted value cant be given as its a point value 
# and prob of point value is zero so find conf interval so prepare 2 equations
# one for lower conf interval (0.025) coefficiants and other with higher(0.975)
# and insert input values in both and find intervals
model_train.summary()
