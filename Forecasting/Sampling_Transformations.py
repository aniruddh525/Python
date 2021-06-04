# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:35:33 2021

@author: anirudh.kumar.verma
"""

#### Upsampling Data

# upsample to daily intervals
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot

sales = read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\sales.csv", header=0, index_col=0, parse_dates=True,squeeze=True)

sales

## upsampling
upsampled = sales.resample('D').mean()
print(upsampled.head(32))

##### interpolate the missing value

interpolated = upsampled.interpolate(method='linear')
print(interpolated.head(32))
interpolated.plot()
sales.plot()

#### Downsampling Data

# downsample to quarterly intervals

resample = sales.resample('Q')
quarterly_mean_sales = resample.mean()
print(quarterly_mean_sales.head())
quarterly_mean_sales.plot()

# Tranformations

airlines = read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\airline-passengers.csv", header=0, index_col=0,parse_dates=True)

# line plot
pyplot.subplot(211) # used to plot both plots in same frame
pyplot.plot(airlines)
# histogram
pyplot.subplot(212)
pyplot.hist(airlines)


#### Square Root Transform


from pandas import DataFrame
from numpy import sqrt


dataframe = DataFrame(airlines.values)
dataframe.columns = ['passengers']
dataframe['passengers'] = sqrt(dataframe['passengers'])

# line plot
pyplot.subplot(211)
pyplot.plot(dataframe['passengers'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['passengers'])


#### Log Transform

from numpy import log
dataframe = DataFrame(airlines.values)
dataframe.columns = ['passengers']
dataframe['passengers'] = log(dataframe['passengers'])

# line plot
pyplot.subplot(211)
pyplot.plot(dataframe['passengers'])
# histogram
pyplot.subplot(212)
pyplot.hist(dataframe['passengers'])


