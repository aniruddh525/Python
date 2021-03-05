# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 09:03:29 2020

@author: anirudh.kumar.verma
"""


#Import Python Libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import seaborn as sns
import scipy.stats as stats
import pylab
import math as mt

#Read csv file
df = pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Basic stat - Level 1\\Q7.csv")

df.mean()
df.median()
df.mode()
df.var()
df.std()
df.Points.max()-df.Points.min()
df.Score.max()-df.Score.min()
df.Weigh.max()-df.Weigh.min()


df = pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Basic stat - Level 1\\Q9_a.csv")

df.skew()
df.kurt()

df.speed.hist()
df.dist.hist()

df.dist.plot(kind="hist")
df.speed.plot(kind="hist")

df = pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Basic stat - Level 1\\Q9_b.csv")

df.skew()
df.kurt()

df.SP.hist()
df.WT.hist()

lst=[34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]
df=pd.DataFrame(lst)

df.mean()
df.median()
df.var()
df.std()

df.hist()

df.skew()
df.kurt()

################# CI ################

xbar=200
s=30
n=2000

##### for 94% CI
xbar-stats.t.ppf(0.97,n-1)*s/mt.sqrt(n) ## 198.737
xbar+stats.t.ppf(0.97,n-1)*s/mt.sqrt(n) ## 201.262

##### for 98% CI
xbar-stats.t.ppf(0.99,n-1)*s/mt.sqrt(n) ## 198.438
xbar+stats.t.ppf(0.99,n-1)*s/mt.sqrt(n) ## 201.561

##### for 96% CI
xbar-stats.t.ppf(0.98,n-1)*s/mt.sqrt(n) ## 198.621
xbar+stats.t.ppf(0.98,n-1)*s/mt.sqrt(n) ## 201.378



cars = pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Basic stat - Level 1\\Cars.csv")

## MPG > 38
1-stats.norm.cdf(38,cars.MPG.mean(),cars.MPG.std())
## 0.3475939251582705

## MPG < 40
stats.norm.cdf(40,cars.MPG.mean(),cars.MPG.std())
## 0.7293498762151616

## 20<MPG<50
stats.norm.cdf(50,cars.MPG.mean(),cars.MPG.std()) - stats.norm.cdf(20,cars.MPG.mean(),cars.MPG.std())
## 0.8988689169682046

######## test to check normal distribution #########

stats.shapiro(cars.MPG)
##(0.9779686331748962, 0.17639249563217163) ---- since P>0.05 so its normal distribution

stats.probplot(cars.MPG,dist='norm',plot=pylab) 

wc_at=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Basic stat - Level 1\\wc-at.csv")
stats.shapiro(wc_at.Waist) ## (0.9558578133583069, 0.001170447445474565) -- Not Normal
stats.shapiro(wc_at.AT) ## (0.9523370265960693, 0.0006539996829815209)  -- Not normal

stats.probplot(wc_at.Waist,dist='norm',plot=pylab) 
stats.probplot(wc_at.AT,dist='norm',plot=pylab)

###### Z scores #####

stats.norm.ppf(0.95) ## 1.644 --- for 90%
stats.norm.ppf(0.97) ## 1.88 -- for 94 %
stats.norm.ppf(0.80) ## 0.841 -- for 60%

###### t values #####

stats.t.ppf(0.975,24) ## 2.063 --- for 95%
stats.t.ppf(0.98,24) ## 2.171 --- for 96%
stats.t.ppf(0.995,24) ## 2.7969 --- for 99%

####################

t=(260-270)/90*mt.sqrt(18) ## t value = -0.47
stats.t.cdf(-0.47,17) # prob = 0.322 ~ 32 %


