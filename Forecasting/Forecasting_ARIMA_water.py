# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 12:12:16 2021

@author: anirudh.kumar.verma
"""

# Import libraries
from pandas import read_csv
from matplotlib import pyplot
from numpy import sqrt
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

water = read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\water.csv", header=0, index_col=0, parse_dates=True)

water

# line plot of time water
water.plot()

water.hist()

water.plot(kind='kde')

# separate out a validation dataset
split_point = len(water) - 10
dataset, validation = water[0:split_point], water[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\My code\\Forecasting\\dataset.csv", header=False)
validation.to_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\My code\\Forecasting\\validation.csv", header=False)


#### Persistence/ Base model

# evaluate a persistence model

from sklearn.metrics import mean_squared_error
from math import sqrt
# load data
train = read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\My code\\Forecasting\\dataset.csv", header=None, index_col=0, parse_dates=True, squeeze=True)
# prepare data
X = train.values
X = X.astype('float32')
train_size = int(len(X) * 0.50)
train, test = X[0:train_size], X[train_size:]


# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    yhat = history[-1]
    predictions.append(yhat)
# observation
    obs = test[i]
    history.append(obs)
    print('>Predicted=%.3f, Expected=%.3f' % (yhat, obs))
# report performance
rmse = sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)

#### ARIMA Hyperparameters


# prepare training dataset


from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARIMA',
                        FutureWarning)


X=X.astype('float32')
train_size=int(len(X)*0.5)
train,test=X[0:train_size],X[train_size:]

model=ARIMA(train,order=(1,1,2)).fit(disp=0)
pred=model.forecast(steps=35)[0]
pred

rmse_arima=sqrt(mean_squared_error(test,pred))
rmse_arima

# evaluate an ARIMA model for a given order (p,d,q) and return RMSE
def evaluate_arima_model(X, arima_order):
# prepare training dataset
    X = X.astype('float32')
    train_size = int(len(X) * 0.50)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]
# make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
# model_fit = model.fit(disp=0)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])
# calculate out of sample error
    rmse = sqrt(mean_squared_error(test, predictions))
    return rmse


#### Grid search for p,d,q values

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float('inf'), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(train, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order,rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))

# load dataset
train = read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\My code\\Forecasting\\dataset.csv", header=None, index_col=0, parse_dates=True, squeeze=True)

# evaluate parameters
p_values = range(0, 5)
d_values = range(1, 5)
q_values = range(0, 5)
warnings.filterwarnings("ignore")
evaluate_models(train.values, p_values, d_values, q_values)

#### Build Model based on the optimized values

# save finalized model to file

# load data
train = read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\My code\\Forecasting\\dataset.csv", header=None, index_col=0, parse_dates=True, squeeze=True)

# prepare data
X = train.values
X = X.astype('float32')

# fit model
model = ARIMA(X, order=(2,1,0))
model_fit = model.fit()
forecast=model_fit.forecast(steps=10)[0]
model_fit.plot_predict(1, 79)

#Error on the test data
val=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\My code\\Forecasting\\validation.csv",header=None)
rmse = sqrt(mean_squared_error(val[1], forecast))
rmse

#### Combine train and test data and build final model

# fit model
# prepare data
X = water.values
X = X.astype('float32')

model = ARIMA(X, order=(2,1,0))
model_fit = model.fit()

forecast=model_fit.forecast(steps=10)[0]
model_fit.plot_predict(1,80)

forecast
