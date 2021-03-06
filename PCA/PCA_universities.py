# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 23:57:00 2021

@author: anirudh.kumar.verma
"""


import pandas as pd 
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
import seaborn as sns 

# load data
universities=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\Universities.csv")
universities.columns
universities.head()

# extract numerical data only
universities1=universities.iloc[:,1:]

# converting DF to array
universities_array=universities1.values

# normalize data (z scores)
universities_norm=scale(universities_array)

# apply model
pcs_model=PCA(n_components=6) # 6 is total no of columns
pcs_values=pcs_model.fit_transform(universities_norm) # columns with pca values
pd.DataFrame(pcs_values)

# variance captured by each PC 
pcs_var = pcs_model.explained_variance_
pd.Series(pcs_var)

# variance captured by each PC 
pcs_var_ratio = (pcs_model.explained_variance_ratio_)*100
pd.Series(pcs_var_ratio)

# calculating cumulative variances
pcs_var_cum=np.cumsum(np.round(pcs_var_ratio,decimals = 4))
pd.Series(pcs_var_cum)


# check if correlation in data has been removed or not

# before applyng PCA
universities1.corr() # high correlation between columns
sns.pairplot(universities1)

# after applyng PCA
pd.DataFrame(pcs_values).corr() # no correlation between columns
sns.pairplot(pd.DataFrame(pcs_values))

# chcek if other condition of sum of squares of weight for any PC is 1 or not.

# PC weights
weights=pcs_model.components_
pd.DataFrame(weights)

# sum of square for  PCs - all are equal to 1
((pd.DataFrame(weights).iloc[:,0])**2).sum()
((pd.DataFrame(weights).iloc[:,1])**2).sum()
((pd.DataFrame(weights).iloc[:,5])**2).sum()
((pd.DataFrame(weights).iloc[:,2])**2).sum()
((pd.DataFrame(weights).iloc[:,3])**2).sum()

# Variance plot for PCA components obtained 
plt.plot(pcs_var_cum,color="red")


# plot between PCA1 and PCA2 
x = pcs_values[:,0:1]
y = pcs_values[:,1:2]
#z = pca_values[:2:3]
plt.scatter(x,y)

# preparing final_df
finalDf = pd.concat([pd.DataFrame(pcs_values[:,0:2],columns=['pc1','pc2']), universities[['Univ']]], axis = 1)

#final scatter plot on final df
sns.scatterplot(data=finalDf,x='pc1',y='pc2',hue='Univ')




