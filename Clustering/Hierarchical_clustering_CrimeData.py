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
grouped_cd


#                Murder     Assault   UrbanPop       Rape  cluster_id
# cluster_id                                                         
# 0            6.055556  140.055556  71.333333  18.683333         0.0
# 1           10.883333  256.916667  78.333333  32.250000         1.0
# 2           10.000000  263.000000  48.000000  44.500000         2.0
# 3           14.671429  251.285714  54.285714  21.685714         3.0
# 4            3.091667   76.000000  52.083333  11.833333         4.0


#Inference
# cluster 2 (Alaska) is highest in crime rate in all aspects inspite of having low population.
# after that we can put cluster 1 though its murder rate is lower than cluster 3.
# then is cluster 3 and after that cluster 0. cluster 4 seems to be safest




