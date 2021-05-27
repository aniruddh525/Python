# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 09:40:30 2020

@author: anirudh.kumar.verma
"""


# Importing necessary libraries
import pandas as pd

# importing data set using pandas
mba = pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\My code\\Data set\\mba.csv")


import scipy.stats as stats
# ppf => Percent point function 
stats.norm.ppf(0.975,0,1)# similar to qnorm in R

# cdf => cumulative distributive function 
stats.norm.cdf(740,711,29) # similar to pnorm in R 

# cummulative distribution function
help(stats.norm.cdf)
#Q-Q plot

import pylab          
import scipy.stats as st

# Checking Whether data is normally distributed
stats.probplot(mba['gmat'], dist="norm",plot=pylab)

stats.probplot(mba.workex,dist="norm",plot=pylab)

mtcars = pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\My code\\Data set\\mtcars.csv")

st.probplot(mtcars.mpg,dist="norm",plot=pylab)
help(st.probplot)


# t distribution 

# Finding qnorm,qt  for 90%,95%,99% confidence level

import scipy.stats as stats
# percentage point function 
stats.norm.ppf(0.975,0,1)# similar to qnorm in R
stats.norm.ppf(0.995,0,1)
stats.norm.ppf(0.950,0,1)
stats.t.ppf(0.975, 139) # similar to qt in R
stats.t.ppf(0.995,139)
stats.t.ppf(0.950,139)
stats.t.cdf(1.65,139)
stats.t.cdf(2.61,139)

help(stats.t.ppf) 

