# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 13:26:34 2021

@author: anirudh.kumar.verma
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# load data
Airlines=pd.read_excel("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Clustering\\EastWestAirlines.xlsx")
Airlines.columns
Airlines.head()

# define normalization function (in 0 to 1)
def norm_func(i):
    return (i-i.min())/(i.max()-i.min())

# Normalize data using function
airline_norm=norm_func(Airlines.iloc[:,1:])

# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(airline_norm, method='average'))

# create clusters and decide no of clusters 
# on the basis of visual clusters obtained form dendogram
clusters = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage = 'average')

# save clusters for chart
cluster_predicted = clusters.fit_predict(airline_norm)
final_Clusters=pd.DataFrame(cluster_predicted,columns=['Clusters'])

# adding cluster_id into original dataframe
Airlines1=Airlines.copy()
Airlines1["cluster_id"]=pd.Series(cluster_predicted)

# grouping based on cluster id
grouped_cd=Airlines1.iloc[:,1:].groupby(Airlines1.cluster_id).mean()
grouped_cd


#Inference
# cluster 2 had only 1 record and it is kind of na outlier mainly vecause of high value of balance.
# cluster 3(high flight trans and bonus trans) and 4 (high CC1 and CC3) are also very small clusters.
# most of the data points are in cluster 0 and 1.

 