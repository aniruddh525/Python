# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 08:46:41 2021

@author: anirudh.kumar.verma
"""

# Importing necessary libraries

import pandas as pd
import scipy.stats as stats
import math as mt

# importing data set using pandas
mba = pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\My code\\Data set\\mba.csv")

stats.norm.interval(0.95)

# to find mean
mu=mba.gmat.mean()
# to find std dev
sig=mba.gmat.std()
# to find sample size
sample_size=len(mba.gmat)
# to chcek if any value is null (two ways)
mba.gmat.isnull().sum()
mba['gmat'].isnull().sum()

# std error
sig/mt.sqrt(sample_size)
# command to find std error
sam_std_err = stats.sem(mba.gmat)
print(sam_std_err)
# command to find CI (similar to CI in R) - when we have sample data only and no pop parameter
stats.t.interval(alpha=0.95,df=sample_size-1,loc=mu,scale=sam_std_err)

# command to find CI (similar to CI in R) - when we have pop parameter
stats.norm.interval(alpha=0.95,loc=mu,scale=sam_std_err)

