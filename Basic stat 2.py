# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 22:34:45 2021

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

####### Set+1_Descriptive+statistics+Probability+(2) ##########

company_data=pd.read_excel("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Basic stat - Level 2\\Dataset.xlsx")
#Boxplot
mpl.pyplot.boxplot(company_data['Measure'])# Morgan Stanley is outlier
company_data.Measure.mean()  ## mean 33.2713
company_data.Measure.var()  ## variance 287.1466
company_data.Measure.std() ## std. dev 16.9454

########### Set+2_Normal+Distribution+Functions+of+random+variables+(1) ######
# 1
1-stats.norm.cdf(50,45,8) # 0.2659
#2
stats.norm.cdf(44,38,6) # less than 44 - 0.841
stats.norm.cdf(44,38,6)-stats.norm.cdf(38,38,6) # between 38 and 44 -- 0.341
stats.norm.cdf(30,38,6) # less than 30 - 0.0912 ~ 9% ~ 36 employess

#4
stats.norm.ppf(0.005,100,20) # 48.48
stats.norm.ppf(0.995,100,20) # 151.516

#5

stats.norm.ppf(0.975,12,5) # 21.8 ~ 981,000,000
stats.norm.ppf(0.025,12,5) # 2.2 ~ 99,000,000

stats.norm.ppf(0.05,12,5) # 3.77 ~ 169,650,000

stats.norm.cdf(0,5,3)  # 0.0477 ~ 4.77%
stats.norm.cdf(0,7,4)  # 0.0400 ~ 4.00%


########### Set 3 Confidence interval########################




########### Set 4 sampling distribution ######

stats.norm.cdf(55,50,4)-stats.norm.cdf(45,50,4) # 0.7887 ~ 78.9%

40/mt.sqrt(250) # 2.52
stats.norm.cdf(55,50,2.52)-stats.norm.cdf(45,50,2.52) # 0.952 ~ 95 %

