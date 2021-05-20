# -*- coding: utf-8 -*-
"""
Created on Thu May 20 10:38:15 2021

@author: anirudh.kumar.verma
"""
# Importing the necessary packages
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold,cross_val_score
from keras.optimizers import Adam
from sklearn.pipeline import Pipeline

# load dataset

gasturbine=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\ANN\\gas_turbines.csv")
gasturbine.columns
gasturbine.head()
gasturbine.info()
gasturbine.describe()

# taking output column at the end
gasturbine=gasturbine[['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT','CDP', 'CO','NOX','TEY']]

gasturbine_arr=gasturbine.values

X = gasturbine_arr[:,0:10]
Y = gasturbine_arr[:,10]


# Standardization
std = StandardScaler()
std.fit(X)
X_std = std.transform(X)
pd.DataFrame(X_std).describe()

# create model
def create_model():
    model = Sequential()
    model.add(Dense(10, input_dim=10, kernel_initializer ='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer ='normal'))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# evaluate model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=create_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=3)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# Standardized: -0.70 (0.10) MSE

# increase no of hidden layers and neurons n each hidden layers

# create model
def increased_model():
    model = Sequential()
    model.add(Dense(20, input_dim=10, kernel_initializer ='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer ='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer ='normal'))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


# evaluate model
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=increased_model, epochs=50, batch_size=5, verbose=0)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=3)
results = cross_val_score(pipeline, X, Y, cv=kfold)
print("Standardized: %.2f (%.2f) MSE" % (results.mean(), results.std()))
# Standardized: -0.92 (0.13) MSE --> MSE has increased so should be tried with other parameters



