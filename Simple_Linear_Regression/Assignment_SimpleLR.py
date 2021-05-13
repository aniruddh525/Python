# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 11:09:13 2021

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

############# Delivery_time  problem #####

Del_time=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Simple Linear Regression\\delivery_time.csv")

#### EDA##

Del_time.columns=['Delivery_Time','Sorting_Time'] # rename columns

Del_time.info() # check for null values

Del_time[Del_time.duplicated()] # to check duplicate rows

Del_time['Delivery_Time'].hist()
Del_time['Sorting_Time'].hist() #### detect outlier
Del_time.boxplot(column='Delivery_Time') #### detect outlier
Del_time.boxplot(column='Sorting_Time')

# check if Linear regression can be aplied

Del_time.corr() # 0.825997
sns.scatterplot(x="Sorting_Time",y="Delivery_Time",data=Del_time)

## Model building

model=smf.ols("Delivery_Time~Sorting_Time",data=Del_time).fit()
model.summary() # Sorting_Time P value = 0.000 and R squared = 0.682
model.params

# reg plot
sns.regplot(x="Sorting_Time", y="Delivery_Time",data=Del_time)


############# Salary_hike  problem #####

Sal_data=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Simple Linear Regression\\Salary_Data.csv")

#### EDA##

Sal_data.info() # check for null values

Sal_data[Sal_data.duplicated()] # to check duplicate rows

Sal_data['Salary'].hist()
Sal_data['YearsExperience'].hist() #### detect outlier
Sal_data.boxplot(column='Salary') #### detect outlier
Sal_data.boxplot(column='YearsExperience')

# check if Linear regression can be aplied

Sal_data.corr() # 0.978242
sns.scatterplot(x="YearsExperience",y="Salary",data=Sal_data)

## Model building

model=smf.ols("Salary~YearsExperience",data=Sal_data).fit()
model.summary() # YearsExperience P value = 0.000 and R squared = 0.957
model.params

# reg plot
sns.regplot(x="YearsExperience", y="Salary",data=Sal_data)


