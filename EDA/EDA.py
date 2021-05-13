# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 22:48:41 2021

@author: anirudh.kumar.verma
"""

import pandas as pd
import numpy as np
import seaborn as sns

Data1 = pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\data_clean.csv")

###### describe data set ##########

Data1.head(10) # first 10 rows
Data1.tail(10) # last 10 rows

Data1.shape ## to get no of rows and columns

Data1.dtypes ## to get data types of each column

Data1.info() ## all info of data --- gives non null value in data set along with datatype

#### removing unwanted column "Unnamed"

data2=Data1.iloc[:,1:]

data=data2.copy() # to copy all values to another variable

data.describe() # gives stats about numeric columns

####### coverting data type #####

data['Month']=pd.to_numeric(data['Month'],errors='coerce') # coerce will introduce
# NA valuesfor non numeric data in the column
data['Temp C']=pd.to_numeric(data['Temp C'],errors='coerce')
## another way to do the same if we are sure which datatype it needs to be converted to
data['Wind']=data['Wind'].astype('int64')

data['Weather']=data['Weather'].astype('category')

data.info()
data.describe()

######### remove duplicates #######

data[data.duplicated()] ## print all duplicate rows

data[data.duplicated()].shape  # count of duplicates

data_cleaned=data.drop_duplicates()
data_cleaned.shape

####### drop columns ########

data_cleaned1=data_cleaned.drop('Temp C',axis=1) # axis =1 is for column and axis =0 for rows

###### rename column #######

data_cleaned2=data_cleaned1.rename({'Solar.R':'Solar'},axis=1)

####### Outlier detection #########

data_cleaned2['Ozone'].hist() #### histogram

data_cleaned2.boxplot(column='Ozone') #### box plot - mpre than 125 is outlier

data_cleaned2[data_cleaned2.Ozone>125] # find index of outlier

data_cleaned2.drop([61,116]) # dropping outlier rows

data_cleaned2['Weather'].value_counts().plot.bar() ### bar plot for categorical data


#### Missing values and Imputation

data_cleaned2.isnull() # to chcek which values are null . if null then TRUE
data_cleaned2.isnull().sum() # True=1 and Flase =0  and then match function is applied

data_cleaned2[data_cleaned2.isnull().any(axis=1)]

cols=data_cleaned2.columns
colours=['#000099','#ffff00'] # yellow = missing , blue = not missing
sns.heatmap(data_cleaned2[cols].isnull(),cmap=sns.color_palette(colours) ) # heatmap

# Mean Imputation

mean_ozone=data_cleaned2['Ozone'].mean()
data_cleaned2['Ozone']=data_cleaned2['Ozone'].fillna(mean_ozone)

# Mode Imputation

obj_columns=data_cleaned2[['Weather']]
obj_columns.isnull().sum()
obj_columns=obj_columns.fillna(obj_columns.mode().iloc[0])

data_cleaned2.mode()

data_cleaned3=pd.concat([data_cleaned2,obj_columns],axis=1) # concatenate to original data frame

data_cleaned3.isnull().sum()

####### to delete a column with duplicate name ######

col_num = [x for x in range(data_cleaned3.shape[1])]
col_num.remove(7)
data_cleaned4=data_cleaned3.iloc[:, col_num]
data_cleaned4.info()


sns.pairplot(data_cleaned4) ### scatter plot #####

data_cleaned4.corr()  ### correlation

####### transformation #########

data_cleaned5=pd.get_dummies(data_cleaned4) # One hot encoding - 
# to clreate dummy variable for categorical variable

data_cleaned5=pd.get_dummies(data_cleaned4,columns=['Weather']) # if only for a specific column







