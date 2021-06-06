# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 12:02:25 2021

@author: anirudh.kumar.verma
"""

## Visualization

# create a line plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

cococola = pd.read_excel("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Forecasting\\CocaCola_Sales_Rawdata.xlsx", header=0, index_col=0,parse_dates=True,squeeze=True)
cococola.plot()

cococola.describe()

#### Histogram and Density Plots

# create a histogram plot
cococola.hist()

# create a density plot
cococola.plot(kind='kde')

#### Box and Whisker Plots by Interval

# create a boxplot of yearly data


# from pandas import Grouper
# groups = cococola.groupby(Grouper(freq='Q'))
# years = pd.DataFrame()
# for name, group in groups:
#     years[name.year] = group.values
# years.boxplot()


#### Lag plot

# create a scatter plot

from pandas.plotting import lag_plot
lag_plot(cococola,lag=1) # for 1st lag. by default its 1

# create an autocorrelation plot

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(cococola,lags=30) # shows for 30 lags


## Data pre processing

cococola = pd.read_excel("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Forecasting\\CocaCola_Sales_Rawdata.xlsx")

# cococola["Date"] = pd.to_datetime(cococola.Quarter,format="%b-%y")
# cococola["month"] = cococola.Date.dt.strftime("%b") # month extraction
# cococola["year"] = cococola.Date.dt.strftime("%Y") # year extraction
# cococola["Day"] = cococola.Date.dt.strftime("%d") # Day extraction
# cococola["wkday"] = cococola.Date.dt.strftime("%A") # weekday extraction

p = cococola["Quarter"][0]
p[0:2]
p[3:]
cococola['Quarters']= 0
cococola['Year']= 0

for i in range(42):
    p = cococola["Quarter"][i]
    cococola['Quarters'][i]= p[0:2]
    

for j in range(42):
    p = cococola["Quarter"][j]
    cococola['Year'][j]= p[3:]

#  create dummy variables for quarters
quarter_pred_dummies = pd.DataFrame(pd.get_dummies(cococola['Quarters']))
cococola = pd.concat([cococola,quarter_pred_dummies],axis = 1)

cococola["t"] = np.arange(1,43)

cococola["t_squared"] = cococola["t"]*cococola["t"]
cococola.columns
cococola["log_sales"] = np.log(cococola["Sales"])

plt.figure(figsize=(12,3))
sns.lineplot(x="Year",y="Sales",data=cococola)

# additional plots

plt.figure(figsize=(12,8))
heatmap_y_quarter = pd.pivot_table(data=cococola,values="Sales",index="Year",columns="Quarters",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_quarter,annot=True,fmt="g") #fmt is format of the grid values


# Boxplot for ever
plt.figure(figsize=(8,6))
plt.subplot(211)
sns.boxplot(x="Quarters",y="Sales",data=cococola)
plt.subplot(212)
sns.boxplot(x="Year",y="Sales",data=cococola)



# Splitting data

Train = cococola.head(38)
Test = cococola.tail(4)

#Linear Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear

#Exponential

Exp = smf.ols('log_sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


#Quadratic 

Quad = smf.ols('Sales~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad

#Additive seasonality 

add_sea = smf.ols('Sales~Q1+Q2+Q3',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1','Q2','Q3']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea

#Additive Seasonality Quadratic 

add_sea_Quad = smf.ols('Sales~t+t_squared+Q1+Q2+Q3',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1','Q2','Q3','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad

##Multiplicative Seasonality

Mul_sea = smf.ols('log_sales~Q1+Q2+Q3',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

#Multiplicative Additive Seasonality 

Mul_Add_sea = smf.ols('log_sales~t+Q1+Q2+Q3',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

#Multiplicative Quadratic Seasonality 

Mul_quad_sea = smf.ols('log_sales~t+t_squared+Q1+Q2+Q3',data = Train).fit()
pred_Mult_quad_sea = pd.Series(Mul_quad_sea.predict(Test))
rmse_Mult_quad_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_quad_sea)))**2))
rmse_Mult_quad_sea 

#Compare the results 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea","rmse_Mult_quad_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea,rmse_Mult_quad_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])

#### Predict for new time period

predict_data = pd.DataFrame()

predict_data["Time Period"]=["Q3-96","Q4-96","Q1-97","Q2-97"]
predict_data["Quarters"]=["Q3","Q4","Q1","Q2"]

Qtr_pred_dummies = pd.DataFrame(pd.get_dummies(predict_data['Quarters']))
predict_data = pd.concat([predict_data,Qtr_pred_dummies],axis = 1)

predict_data["t"] = np.arange(43,47)
predict_data["t_squared"] = predict_data["t"]*predict_data["t"]
predict_data.columns



#Build the model on entire data set
model_full = smf.ols('log_sales~t+Q1+Q2+Q3',data = cococola).fit()

pred_new  = pd.Series(model_full.predict(predict_data))
pred_new

predict_data["forecasted_Sales"] = pd.Series(round(np.exp(pred_new)))

predict_data


##### Data driven models #####

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Moving Average (centered Moving average)

plt.figure(figsize=(12,4))
cococola.Sales.plot(label="org")
for i in range(2,24,6):
    cococola["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')

# Time series decomposition plot 


decompose_ts_add = seasonal_decompose(cococola.Sales,freq=12)
decompose_ts_add.plot()


# ACF plots and PACF plots


import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cococola.Sales,lags=12)
tsa_plots.plot_pacf(cococola.Sales,lags=12)



### Evaluation Metric MAPE

def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)

### Simple Exponential Method


ses_model = SimpleExpSmoothing(Train["Sales"]).fit(smoothing_level=0.9)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales) 


### Holt method 

# Holt method 
hw_model = Holt(Train["Sales"]).fit(smoothing_level=0.8, smoothing_trend=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales) 

hw_model.summary()
### Holts winter exponential smoothing with additive seasonality and additive trend


hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=4).fit() #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Sales) 
hwe_model_add_add.summary()
### Holts winter exponential smoothing with multiplicative seasonality and additive trend

hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=4).fit() 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Sales)
hwe_model_mul_add.summary()
## Final Model by combining train and test

hwe_model_add_add = ExponentialSmoothing(cococola["Sales"],seasonal="mul",trend="add",seasonal_periods=4).fit()
hwe_model_add_add.summary()
#Forecasting for next 10 time periods
hwe_model_add_add.forecast(4)















