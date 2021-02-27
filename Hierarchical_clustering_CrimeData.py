# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 00:44:52 2021

@author: anirudh.kumar.verma
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# load data
crime_data=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Clustering\\crime_data.csv")
crime_data.columns

crime_data.head()

# define normalization function (in 0 to 1)
def norm_func(i):
    return (i-i.min())/(i.max()-i.min())

# Normalize data using function
cd_norm=norm_func(crime_data.iloc[:,1:])

# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(cd_norm, method='average'))

# create clusters and decide no of clusters 
# on the basis of visual clusters obtained form dendogram
clusters = AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage = 'average')

# save clusters for chart
cluster_predicted = clusters.fit_predict(cd_norm)
final_Clusters=pd.DataFrame(cluster_predicted,columns=['Clusters'])

# adding cluster_id into original dataframe
crime_data["cluster_id"]=pd.Series(cluster_predicted)

# grouping based on cluster id
grouped_cd=crime_data.groupby(crime_data.cluster_id).mean()

