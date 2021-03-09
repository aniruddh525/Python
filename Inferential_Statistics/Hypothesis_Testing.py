# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 18:53:45 2021

@author: anirudh.kumar.verma
"""


##### 1 sample T test ####### 
## similar to "t.test(sample_data,alternative = 'greater',mu=0.3)" in R

import pandas as pd
import numpy as np
import scipy 
from scipy import stats
import statsmodels.api as sm

data=pd.Series([0.593,0.142,0.329,0.691,0.231,0.793,0.519,0.392,0.418])

p=scipy.stats.ttest_1samp(data,0.3)[1] # [1] is used since 2nd value is prob value
# p value produced above is two tail test p value so divide by 2 to get final p value
p_val=p/2
round(p_val,5)

##### 2 sample T test ####### 
## similar to "t.test(control_grp,Treat_grp,alternative = 'two.sided')" in R

control = pd.Series([91,87,99,77,88,91])
Treat = pd.Series([101,110,103,93,99,104])

scipy.stats.ttest_ind(control, Treat)

## Ttest_indResult(statistic=-3.445612673536487, pvalue=0.006272124350809803) since P<0.05
# so reject Ho and MU1 is not equal to MU2

######### 2 proportion test ###### used for categorical variable ######

n1=247
p1=0.37

n2=308
p2=0.39

pop1=np.random.binomial(1,p1,n1)
pop1.mean()
pop2=np.random.binomial(1,p2,n2)
pop2.mean()

scipy.stats.ttest_ind(pop1, pop2) #or the other way is from statsmodels.api

sm.stats.ttest_ind(pop1, pop2)

############# Anova #####################################################

# importing data set using pandas
anv_data = pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\ContractRenewal_Data(unstacked).csv")
anv_data.columns='supA','supB','supC'

stats.f_oneway(anv_data.supA,anv_data.supB,anv_data.supC)

################Chi-Square Test ### to test independence between 2 variables H0 -> ind. Ha -> dep #############

chi_data=[[14,4],[0,10]]
stats.chi2_contingency(chi_data) # p value = 0.000385

