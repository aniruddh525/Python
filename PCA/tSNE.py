# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 12:48:26 2021

@author: anirudh.kumar.verma
"""

from pandas import read_csv
import pandas as pd

from sklearn.manifold import TSNE
from bioinfokit.visuz import cluster

# load data
filename = "C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\TSNE_data.csv"
dataframe = pd.read_csv(filename)

# Split-out validation dataset
array = dataframe.values
# separate array into input and output components
X = array[:,1:]
Y = array[:,0]

#TSNE visualization

data_tsne = TSNE(n_components=2).fit_transform(X)
cluster.tsneplot(score=data_tsne)

# get a list of categories
color_class = dataframe['diagnosis'].to_numpy()
cluster.tsneplot(score=data_tsne, colorlist=color_class, legendpos='upper right',legendanchor=(1.15, 1))

#Plot will be stored in the default directory

data_tsne

