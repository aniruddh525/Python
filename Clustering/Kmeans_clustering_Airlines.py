# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:45:35 2021

@author: anirudh.kumar.verma
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# load data
Airlines=pd.read_excel("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Clustering\\EastWestAirlines.xlsx")
Airlines.columns
Airlines.head()

# define normalization function (in 0 to 1)
def norm_func(i):
    return (i-i.min())/(i.max()-i.min())

# Normalize data using function
airline_norm=norm_func(Airlines.iloc[:,1:])

# finding optimal K value using scree plot or elbow curve 
k = list(range(2,15))
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(airline_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(airline_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,airline_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

WSS
TWSS

# Scree plot 
plt.plot(k,TWSS, 'ro-')
plt.xlabel("No_of_Clusters")
plt.ylabel("total_within_SS")
plt.xticks(k)

# Selecting 5 clusters as 5 looks optimal from the above scree plot 
model=KMeans(n_clusters=5) 
model.fit(airline_norm)

model.labels_ 
md=pd.Series(model.labels_)   

Airlines1=Airlines.copy()
Airlines1['cluster_id']=md # creating a  new column for assigning cluster values

grouped_clusters=Airlines1.iloc[:,1:].groupby(Airlines1.cluster_id).mean()
grouped_clusters


# cluster_id	Balance	Qual_miles	cc1_miles	cc2_miles	cc3_miles	Bonus_miles	Bonus_trans	Flight_miles_12mo	Flight_trans_12	Days_since_enroll	Award?	cluster_id
# 0	33097.301356589145	94.13178294573643	1.070736434108527	1.0164728682170543	1.0067829457364341	3244.5203488372094	6.1734496124031	212.85077519379846	0.6036821705426356	1992.4021317829458	0.0	0
# 1	108317.38737623762	198.33663366336634	3.9158415841584158	1.0012376237623761	1.025990099009901	45609.65717821782	20.201732673267326	713.7289603960396	2.142326732673267	4863.439356435643	1.0	1
# 2	83529.1530460624	290.45319465081724	1.1560178306092124	1.0326894502228827	1.0089153046062407	8850.395245170877	10.476968796433878	1030.112927191679	3.148588410104012	4338.867756315008	1.0	2
# 3	118297.32524271845	73.46763754045307	3.5841423948220066	1.0016181229773462	1.022653721682848	31384.393203883494	17.233009708737864	224.10032362459546	0.627831715210356	4419.553398058252	0.0	3
# 4	49921.633640552995	89.90322580645162	1.1221198156682028	1.0195852534562213	1.0011520737327189	3467.0748847926266	6.913594470046083	243.83410138248848	0.728110599078341	5567.925115207373	0.0	4

#Inference
# cluster 0 is the cluster of customers who has joined last and thats why most of the fields have less values.
# cluster 4 is the cluster who has joined the program first but still asstes erned are not that much.
# cluster 1 is the cluster who joined 2nd (after cluster 4) , majority of assets are high.

