# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:04:53 2021

@author: anirudh.kumar.verma
"""

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# load data
crime_data=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Clustering\\crime_data.csv")
crime_data.columns
crime_data.head()

# define normalization function (in 0 to 1)
def norm_func(i):
    return (i-i.min())/(i.max()-i.min())

# Normalize data using function
cd_norm=norm_func(crime_data.iloc[:,1:])

# finding optimal K value using scree plot or elbow curve 
k = list(range(2,8))
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(cd_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(cd_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,cd_norm.shape[1]),"euclidean")))
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
model.fit(cd_norm)

model.labels_ 
md=pd.Series(model.labels_)   

crime_data1=crime_data.copy()
crime_data1['cluster_id']=md # creating a  new column for assigning cluster values

grouped_clusters=crime_data1.groupby(crime_data1.cluster_id).mean()
grouped_clusters


#                Murder     Assault   UrbanPop       Rape
# cluster_id                                             
# 0           10.966667  264.000000  76.500000  33.608333
# 1            2.680000   70.100000  51.000000  10.910000
# 2           14.671429  251.285714  54.285714  21.685714
# 3            4.428571  129.000000  82.000000  17.685714
# 4            6.950000  143.357143  63.928571  19.542857

#Inference
# cluster 0 is highest in crime rate except murder.
# then comes cluster 2 , then 4 , then 3. 1 is safest of all


