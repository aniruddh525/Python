# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 19:23:51 2021

@author: anirudh.kumar.verma
"""


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# load data
Universities=pd.read_csv('C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\Universities.csv')
Universities.columns
Universities.head()

# define normalization function (in 0 to 1)

def norm_func(i):
    return (i-i.min())/(i.max()-i.min())

# Normalize data using function
univ_norm=norm_func(Universities.iloc[:,1:])


# perform clustering for K =3
kmeans_cluster=KMeans(n_clusters=3)
kmeans_cluster.fit(univ_norm)
kmeans_cluster.labels_
kmeans_cluster.cluster_centers_3
Universities1=Universities.copy()
Universities1["cluster_id"]=pd.Series(kmeans_cluster.labels_)
grouped_clusters=Universities1.groupby(Universities1.cluster_id).mean()
grouped_clusters


# finding optimal K value
# for smaller data set , thumb rule is k = sqrt(n)/2 where n = no od Data points
# for large data set, we use elbow / scree plot

# screw plot or elbow curve ############
k = list(range(2,15))
k
TWSS = [] # variable for storing total within sum of squares for each kmeans 
for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(univ_norm)
    WSS = [] # variable for storing within sum of squares for each cluster 
    for j in range(i):
        WSS.append(sum(cdist(univ_norm.iloc[kmeans.labels_==j,:],kmeans.cluster_centers_[j].reshape(1,univ_norm.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))

WSS
TWSS

# Scree plot 
plt.plot(k,TWSS, 'ro-')
plt.xlabel("No_of_Clusters")
plt.ylabel("total_within_SS")
plt.xticks(k)

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=5) 
model.fit(univ_norm)

model.labels_ # getting the labels of clusters assigned to each row 
md=pd.Series(model.labels_)  # converting numpy array into pandas series object 
Universities['clust']=md # creating a  new column and assigning it to new column 
univ_norm.head()
Universities1=Universities.copy()
grouped_clusters=Universities1.groupby(Universities1.clust).mean()
grouped_clusters

















