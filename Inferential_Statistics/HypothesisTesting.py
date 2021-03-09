# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:36:12 2021

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
from statsmodels.stats.proportion import proportions_ztest

######### Cutlet ###########

Cutlets=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Hypothesis Testing\\Cutlets.csv")
Cutlets.columns =['Unit_A','Unit_B']

# Ho -> MUa=MUb    Ha -> MUa<>MUb     Conducting 2 sample T test

scipy.stats.ttest_ind(Cutlets.Unit_A,Cutlets.Unit_B) # statistic=0.7228688704678061, 
#pvalue=0.4722394724599501------- since P > 0.05 so we dont reject Ho and both units are same


######## LabTAT ########

Lab_TAT=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Hypothesis Testing\\LabTAT.csv")
Lab_TAT.columns =['Lab_1','Lab_2','Lab_3','Lab_4']

###### applying f_oneway for Anova ################

stats.f_oneway(Lab_TAT.Lab_1,Lab_TAT.Lab_2,Lab_TAT.Lab_3,Lab_TAT.Lab_4)
# statistic=118.70421654401437, pvalue=2.1156708949992414e-57 
# since p < 0.05 , so we reject Ho that means atleast one lab average TAT is difefrent.

########## Buyer Ratio ################

Buyer_Ratio=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Hypothesis Testing\\BuyerRatio.csv")

# Males=np.array(Buyer_Ratio.loc[0,['East','West','North','South']])
# Females=np.array(Buyer_Ratio.loc[1,['East','West','North','South']])

Buyer_data=Buyer_Ratio.values[:,[1,2,3,4]].tolist()
stats.chi2_contingency(Buyer_data) # p value = 0.6603 since p>0.05 so Ho is not rejected.

########## Customer Order ################

Cust_Order=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Hypothesis Testing\\Costomer+OrderForm.csv")


Cust_data=Cust_Order.apply(pd.value_counts).values.tolist()
stats.chi2_contingency(Cust_data) # p value = 0.27710 since p>0.05 so Ho is not rejected

############### Faltoons ################################

Faltoons=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Hypothesis Testing\\Faltoons.csv")
Faltoons_data=Faltoons.apply(pd.value_counts).values.tolist()

stats.chi2_contingency(Faltoons_data) # p value = 8.54342267020237e-05 since p<0.05 so Ho rejected
# and managers comment is correct.

