# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:14:28 2021

@author: anirudh.kumar.verma
"""

#Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Load data
wholesale_cust=pd.read_csv('C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\Wholesale customers data.csv')
wholesale_cust.columns
wholesale_cust.head()
wholesale_cust.info()

# drop columns which are not required for clustering
wholesale_cust1=wholesale_cust.copy()
wholesale_cust1.drop(['Channel','Region'],axis=1,inplace=True)

# extract data in array to apply transformation
cust_array=wholesale_cust1.values

# convert to std values (z values)
stscalar=StandardScaler().fit(cust_array)
x=stscalar.transform(cust_array)

# build model
model=DBSCAN(eps=0.3,min_samples=7)
model.fit(x)

# check clusters

pd.Series(model.labels_)
pd.Series(model.labels_).value_counts()

# -1    333
#  0     99
#  1      8

# label -1 are given to outliers , 333 are outliers out of 440 
#so that means parameters are not correct so try wit other parameters

model=DBSCAN(eps=0.5,min_samples=9)
model.fit(x)
pd.Series(model.labels_).value_counts()

# add clusters to main data
wholesale_cust1["cluster_id"]=pd.DataFrame(model.labels_)

grouped_data=wholesale_cust1.groupby(wholesale_cust1.cluster_id).mean()

