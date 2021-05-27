# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 21:32:54 2021

@author: anirudh.kumar.verma
"""

import pandas as pd
import numpy as np
import scipy 
from scipy import stats
import statsmodels.stats.descriptivestats as sd
import matplotlib as mpl
import seaborn as sns
import math as mt
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

############# Newspaper problem #####

Nd=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\NewspaperData.csv")

### check if their is any null values and any cleaning required
Nd.info()

# chcek if Linear regression can be aplied

Nd.corr()
sns.scatterplot(x="daily",y="sunday",data=Nd)

# chcek if data is normally distributed or not 

sns.distplot(Nd["daily"])
sns.distplot(Nd["sunday"])

# model building

model=smf.ols("sunday~daily",data=Nd).fit()
model.summary()
model.params

# reg plot

sns.regplot(x="daily", y="sunday",data=Nd)

# prediction
newdata=pd.Series([200,300])
data_pred=pd.DataFrame(newdata,columns=['daily'])
model.predict(data_pred)



########## WC - Adipose tissue problem #######

Wc_AT=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\WC_AT.csv")

Wc_AT.info()

Wc_AT.corr()
sns.scatterplot(x="Waist",y="AT",data=Wc_AT)
sns.distplot(Wc_AT["Waist"])
sns.distplot(Wc_AT["AT"])

model=smf.ols("AT~Waist",data=Wc_AT).fit()
model.summary()
model.params

##### another method using scikit-learn ####


x=np.array(Wc_AT.Waist).reshape(-1,1) # x variable should be a 2D array so coverting that
sk_model = LinearRegression().fit(x, Wc_AT.AT) # model building
sk_model.score(x, Wc_AT.AT)  # r square value
sk_model.intercept_ # beta not
sk_model.coef_ # beta one( slope)

x_new=[[80],[121]] # input should be a 2D array
y_pred=sk_model.predict(x_new)

