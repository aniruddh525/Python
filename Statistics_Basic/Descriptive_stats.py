# -*- coding: utf-8 -*-
"""
Created on Fri Dec 11 09:08:03 2020

@author: anirudh.kumar.verma
"""


#Import Python Libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import seaborn as sns

#Read csv file
df = pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\My code\\Data set\\mba.csv")

#Read first 5 rows
df.head()

#Read last 5 rows
df.tail()

#Check a particular column type
df['gmat'].dtype

#Check types for all the columns
df.dtypes

#return max/min values for all numeric columns

df.max()
df.min()

#return mean/median values for all numeric columns

df.mean()
df.median()

#standard deviation

df.std()

# returns a random sample of the data frame

## df.sample([n=2]) - to check correct usage

# drop all the records with missing values

df_na = pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\My code\\Data set\\mba_na.csv")

df_na.dropna()

#Selecting a column in a Data Frame
df['workex']
df.workex

#to get count of values

df.gmat.count()
df.count()

#Group data using class
df_class = df.groupby(['class'])
df_class

df_class.mean()
df_class.gmat.mean()

df_gmat=df.groupby(['gmat'])
df_gmat.mean()

# in double brackets , out generated is data frame otherwise panads series obj
df.groupby('class')[['workex']].mean() 

df.groupby(['class'],sort=False)[['workex']].mean()

# Data frame filter

df_sub=df[ df['gmat'] > 700 ]
df_sub
df_class_A=df[df['class']=='A']

# Data frame slicing

df_sliced=df[['gmat','class']]


#Select rows by their position:

df[10:20]

#Select rows by their labels:

df_sub.loc[10:20,['gmat','class']]

df_sub.iloc[10:20,[0, 3]]

# iloc function

df.iloc[[0]]

df.iloc[:,2]

df.iloc[1:5, 0:2]  
df.iloc[[0,5], [1,3]]  

# Create a new data frame from the original sorted by the column workex
df_sorted = df.sort_values( by ='workex')
df_sorted.head()

# sorting using 2 columns

df_sorted_2 = df.sort_values( by =['workex', 'gmat'], ascending = [True, False])
df_sorted_2.head(10)

# Select the rows that have at least one missing value

df_na[df_na.isnull().any(axis=1)]

df.var()
df.std()
df.mode()

# aggregate functions

df[['workex','gmat']].agg(['min','mean','max','std'])

df.describe()
df.skew()
df.kurt()

# graphs

import matplotlib.pyplot as plt 

plt.plot([0,2,8,4,10])
plt.plot([1,2,3,4,5], [1,2,3,4,10]) 
plt.plot([1,2,3,4,5], [1,2,3,4,10],'go') 
plt.show()

plt.plot([1,2,3,4,5], [1,2,3,4,10], 'go') # green dots 
plt.plot([1,2,3,4,5], [2,3,4,5,11], 'b*') # blue stars


y = [3, 10, 7, 5, 3, 4.5, 6, 8.1]
N = len(y)
x = range(N)
width = 1/1.5
plt.bar(x, y, width, color="blue")

plt.plot([1,2,3,4,5], [1,2,3,4,10], 'go', label='GreenDots')
plt.plot([1,2,3,4,5], [2,3,4,5,11], 'b*', label='Bluestars') 
plt.title('A Simple Scatterplot') 
plt.xlabel('X') 
plt.ylabel('Y') 
plt.legend(loc='best') # legend text comes from the plot's label parameter. 
plt.show()

plt.figure(figsize=(10,7)) # 10 is width, 7 is height plt.plot([1,2,3,4,5], [1,2,3,4,10], 'go', label='GreenDots') # green dots 
plt.plot([1,2,3,4,5], [2,3,4,5,11], 'b*', label='Bluestars') # blue stars 
plt.title('A Simple Scatterplot')
plt.xlabel('X') 
plt.ylabel('Y') 
plt.xlim(0, 6) 
plt.ylim(0, 12) 
plt.legend(loc='best') 
plt.show()

# Graphical Representation of data

mba = pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\My code\\Data set\\mba.csv")
import matplotlib.pylab as plt
# Histogram
plt.hist(mba['gmat']) # left skew 

#Boxplot
plt.boxplot(mba['gmat'])# for vertical
plt.boxplot(mba['gmat'],1,'rs',0)# For Horizontal
help(plt.boxplot)

# Barplot
# bar plot we need height i.e value of each data
# left - for starting point of each bar plot of data on X-axis(Horizontal axis). Here data is mba['gmat']
index = np.arange(773) # np.arange(a)  = > creates consecutive numbers from 0 to 772 

mba.shape # dimensions of data frame 
# below code not working 
# plt.bar(x,height = mba["gmat"], left = np.arange(1,774,1)) # initializing the parameter 
# # left with index values 
# help(plt.bar)

import pandas as pd
import matplotlib.pyplot as plt
mtcars = pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\My code\\Data set\\mtcars.csv")

# table 
pd.crosstab(mtcars.gear,mtcars.cyl)

# bar plot between 2 different categories 
pd.crosstab(mtcars.gear,mtcars.cyl).plot(kind="bar")

import seaborn as sns 
# getting boxplot of mpg with respect to each category of gears 
sns.boxplot(x="gear",y="mpg",data=mtcars)

sns.pairplot(mtcars.iloc[:,0:3]) # histogram of each column and 
# scatter plot of each variable with respect to other columns 

import numpy as np
plt.plot(np.arange(32),mtcars.mpg) # scatter plot of single variable

plt.plot(np.arange(32),mtcars.mpg,"ro-")

help(plt.plot) # explore different visualizations among the scatter plot

# Scatter plot between different inputs

plt.plot(mtcars.mpg,mtcars["hp"],"ro");plt.xlabel("mpg");plt.ylabel("hp")

mtcars.hp.corr(mtcars.mpg)
mtcars.corr()
# ro  indicates r - red , o - points 
# 
# group by function 
mtcars.mpg.groupby(mtcars.gear).median() # summing up all mpg with respect to gear

mtcars.cyl.value_counts()
# pie chart
mtcars.gear.value_counts().plot(kind="pie")

mtcars.mpg.groupby(mtcars.gear).plot(kind="line")
mtcars.gear.plot(kind="pie")
# bar plot for count of each category for gear 
mtcars.gear.value_counts().plot(kind="bar")


# histogram of mpg for each category of gears 
mtcars.mpg.groupby(mtcars.gear).plot(kind="hist") 
mtcars.mpg.groupby(mtcars.gear).count()

# line plot for mpg column
mtcars.mpg.plot(kind='line') 
plt.plot(np.arange(32),mtcars.mpg,"ro-")


mtcars.mpg = mtcars.mpg.astype(str)
mtcars.gear = mtcars.gear.astype(str)
mtcars.groupby(mtcars.gear).count()

mtcars.groupby("gear")["mpg"].apply(lambda x: x.mean())

mtcars.gear.value_counts().plot(kind="pie")
mtcars.gear.value_counts().plot(kind="bar")

mtcars.head()

pd.crosstab(mtcars.gear,mtcars.cyl).plot(kind="bar",stacked=False,grid=True)
plt.scatter(mtcars.mpg,mtcars.wt)
mtcars.plot(kind="scatter",x="mpg",y="wt")
mtcars.mpg.plot(kind="hist")

import seaborn as sns
# sns.pairplot(mtcars.mpg,hue="gear",size=3,diag_kind = "kde")

sns.FacetGrid(mtcars,hue="cyl").map(plt.scatter,"mpg","wt").add_legend()
sns.boxplot(x="cyl",y="mpg",data=mtcars)
sns.FacetGrid(mtcars,hue="cyl").map(sns.kdeplot,"mpg").add_legend()

sns.pairplot(mtcars,hue="gear",size=3)




