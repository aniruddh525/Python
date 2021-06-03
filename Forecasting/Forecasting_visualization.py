# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 12:01:25 2021

@author: anirudh.kumar.verma
"""

## Line plot

# create a line plot
from pandas import read_csv
from matplotlib import pyplot

series = read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\daily-minimum-temperatures.csv", header=0, index_col=0,parse_dates=True)
series.plot()

series.describe()

#### Histogram and Density Plots

# create a histogram plot
series.hist()

# create a density plot
series.plot(kind='kde')
pyplot.show()

#### Box and Whisker Plots by Interval

# create a boxplot of yearly data

from pandas import DataFrame
from pandas import Grouper
series = read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\daily-minimum-temperatures.csv", header=0, index_col=0,parse_dates=True,squeeze=True)      
groups = series.groupby(Grouper(freq='A'))
years = DataFrame()
for name, group in groups:
    years[name.year] = group.values
years.boxplot()


#### Lag plot

# create a scatter plot

from pandas.plotting import lag_plot
lag_plot(series,lag=1) # for 1st lag. by default its 1

# create an autocorrelation plot

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(series,lags=30) # shows for 30 lags
