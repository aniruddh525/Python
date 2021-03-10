# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:21:47 2021

@author: anirudh.kumar.verma
"""
import pandas as pd 
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation

# load data
book=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Recommendation Engine\\book.csv",encoding = "ISO-8859-1")
book.columns
book.head()
book.info()

book=book.iloc[:,1:]

book.rename(columns = {'User.ID':'User_ID', 'Book.Title':'Book_Title', 'Book.Rating':'Book_Rating'}, inplace = True) 

#number of unique users in the dataset
len(book.User_ID.unique())
len(book.Book_Title.unique())

# convert data into n * p matrix
Book_pivot = book.pivot_table(index='User_ID',
                                 columns='Book_Title',
                                 values='Book_Rating').reset_index(drop=True)

# replace index of pivot with user id values
Book_pivot.index = book.User_ID.unique()

#Impute NaNs with 0 values
Book_pivot.fillna(0, inplace=True)

# calculate distance 
# pairwise gives (1-cos(A,B)) so to negate that we are using 1- ahead of that
user_distance = 1 - pairwise_distances( Book_pivot.values,metric='cosine')

distance_matrix=pd.DataFrame(user_distance)

#Set the index and column names to user ids 
distance_matrix.index = book.User_ID.unique()
distance_matrix.columns = book.User_ID.unique()

# fill diagnols with 0s as diagnol has all 1

np.fill_diagonal(distance_matrix.values,0)

#Most Similar Users
distance_matrix.idxmax(axis=1)[0:5]



