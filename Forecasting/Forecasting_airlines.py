# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 22:47:29 2021

@author: anirudh.kumar.verma
"""

## Visualization

# create a line plot
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

airline = pd.read_excel("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Forecasting\\Airlines_Data.xlsx", header=0, index_col=0,parse_dates=True,squeeze=True)
airline.plot()

airline.describe()

#### Histogram and Density Plots

# create a histogram plot
airline.hist()

# create a density plot
airline.plot(kind='kde')

#### Box and Whisker Plots by Interval

# create a boxplot of yearly data


from pandas import Grouper
groups = airline.groupby(Grouper(freq='A'))
years = pd.DataFrame()
for name, group in groups:
    years[name.year] = group.values
years.boxplot()


#### Lag plot

# create a scatter plot

from pandas.plotting import lag_plot
lag_plot(airline,lag=1) # for 1st lag. by default its 1

# create an autocorrelation plot

from statsmodels.graphics.tsaplots import plot_acf
plot_acf(airline,lags=30) # shows for 30 lags


## Data pre processing

airline = pd.read_excel("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Forecasting\\Airlines_Data.xlsx")

airline["Date"] = pd.to_datetime(airline.Month,format="%b-%y")
airline["month"] = airline.Date.dt.strftime("%b") # month extraction
airline["year"] = airline.Date.dt.strftime("%Y") # year extraction
airline["Day"] = airline.Date.dt.strftime("%d") # Day extraction
airline["wkday"] = airline.Date.dt.strftime("%A") # weekday extraction

plt.figure(figsize=(12,8))
heatmap_y_month = pd.pivot_table(data=airline,values="Passengers",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") #fmt is format of the grid values


# Boxplot for ever
plt.figure(figsize=(8,6))
plt.subplot(211)
sns.boxplot(x="month",y="Passengers",data=airline)
plt.subplot(212)
sns.boxplot(x="year",y="Passengers",data=airline)

# Create dummy variables
import numpy as np   
month_dummies = pd.DataFrame(pd.get_dummies(airline['month']))
airline1 = pd.concat([airline,month_dummies],axis = 1)

airline1["t"] = np.arange(1,97)

airline1["t_squared"] = airline1["t"]*airline1["t"]
airline1.columns
airline1["log_pass"] = np.log(airline1["Passengers"])

plt.figure(figsize=(12,3))
sns.lineplot(x="year",y="Passengers",data=airline1)

# Splitting data

Train = airline1.head(84)
Test = airline1.tail(12)

#Linear Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear

#Exponential

Exp = smf.ols('log_pass~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


#Quadratic 

Quad = smf.ols('Passengers~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad

#Additive seasonality 

add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea

#Additive Seasonality Quadratic 

add_sea_Quad = smf.ols('Passengers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad

##Multiplicative Seasonality

Mul_sea = smf.ols('log_pass~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

#Multiplicative Additive Seasonality 

Mul_Add_sea = smf.ols('log_pass~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

#Multiplicative Quadratic Seasonality 

Mul_quad_sea = smf.ols('log_pass~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_quad_sea = pd.Series(Mul_quad_sea.predict(Test))
rmse_Mult_quad_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_quad_sea)))**2))
rmse_Mult_quad_sea 




#Compare the results 

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea","rmse_Mult_quad_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea,rmse_Mult_quad_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])

#### Predict for new time period

predict_data = pd.DataFrame()

predict_data["Time Period"]=["Jan-03","feb-03","Mar-03","Apr-03",
                             "May-03","Jun-03","Jul-03","Aug-03",
                             "Sep-03","Oct-03","Nov-03","Dec-03"]

p = predict_data["Time Period"][0]
p[0:3]
predict_data['months']= 0

for i in range(12):
    p = predict_data["Time Period"][i]
    predict_data['months'][i]= p[0:3]


month_pred_dummies = pd.DataFrame(pd.get_dummies(predict_data['months']))
predict_data = pd.concat([predict_data,month_pred_dummies],axis = 1)

predict_data["t"] = np.arange(97,109)

predict_data["t_squared"] = predict_data["t"]*predict_data["t"]
predict_data.columns
predict_data.rename({'feb':"Feb"},axis=1,inplace=True)


#Build the model on entire data set
model_full = smf.ols('log_pass~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = airline1).fit()

pred_new  = pd.Series(model_full.predict(predict_data))
pred_new

predict_data["forecasted_Passengers"] = pd.Series(round(np.exp(pred_new)))

predict_data















