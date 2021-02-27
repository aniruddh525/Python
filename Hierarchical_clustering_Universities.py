# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 23:14:24 2021

@author: anirudh.kumar.verma
"""

# import hierarchical clustering libraries

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# load data
Universities=pd.read_csv('C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\Universities.csv')
Universities.columns

Universities.head()

# define normalization function (in 0 to 1)

def norm_func(i):
    return (i-i.min())/(i.max()-i.min())

# Normalize data using function
univ_norm=norm_func(Universities.iloc[:,1:])

# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(univ_norm, method='average'))

# create clusters and decide no of clusters 
# on the basis of visual clusters obtained form dendogram
clusters = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'average')

# save clusters for chart
cluster_predicted = clusters.fit_predict(univ_norm)
final_Clusters=pd.DataFrame(cluster_predicted,columns=['Clusters'])

# adding cluster_id into original dataframe
Universities["cluster_id"]=pd.Series(cluster_predicted)

# grouping based on cluster id
grouped_univ=Universities.groupby(Universities.cluster_id).mean()




















