import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf 

cola=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\CocaCola_Sales_Rawdata.csv")
cola.columns
cola["Sales"].plot(figsize=(12,7)) # seems like linear upward trend and additive seasonality
cola.rename(columns={'Quarter':'Quarter_Year'},inplace=True)
cola["Quarter_Year"][0][3:]
type(cola["Quarter_Year"][0][3:])
int(cola["Quarter_Year"][0][3:])+1900

cola["Quarter"]=0
cola["Year"]=0

for i in range(42):
    cola["Quarter"][i]=cola["Quarter_Year"][i][:2]
    cola["Year"][i]=int(cola["Quarter_Year"][i][3:])+1900


mean_sales=[]
for i in range(0,40,4):
    avg=(cola["Sales"][i]+cola["Sales"][i+1]+cola["Sales"][i+2]+cola["Sales"][i+3])/4
    mean_sales.append(avg)
    
x=np.arange(10)
plt.plot(x,mean_sales)
cola.Sales.rolling(4).mean().plot() # not perfectly linear

sns.distplot(cola.Sales,bins=10,kde=True)  # positive skew ie more instances of lower sales
cola.Sales.skew()  #0.630
cola.Sales.kurt()  #-0.58  # no peakedness

sns.boxplot(x="Quarter",y="Sales",data=cola)  #more sales seen in quarter 2
sns.boxplot(x="Year",y="Sales",data=cola)    # least sales in 1987,1989 also presence of outliers to the lower extreme 

sns.lineplot(x="Quarter",y="Sales",hue="Year",data=cola) # sales are increasing by the years
sns.lineplot(x="Year",y="Sales",hue="Quarter",data=cola) # sales less in Q1 and Q4, higher in Q2 and Q3

cola["t"]=np.arange(1,43)
cola["t_sq"]=cola["t"]*cola["t"]
cola["log_Sales"]=np.log(cola["Sales"])
q_dummies=pd.get_dummies(cola['Quarter']) # 4 dummy variables for quarters
cola=pd.concat([cola,q_dummies],axis=1)

train=cola.iloc[:32,:]
test=cola.iloc[32:,:]

#####   Model Based Techniques   #####

#  -->Linear Model
lin_model=smf.ols('Sales~t',data=train).fit()
lin_pred=lin_model.predict(test['t'])
lin_rmse=np.sqrt(np.mean((lin_pred-test['Sales'])**2))

plt.rcParams.update({'figure.figsize':(8,5),'figure.dpi':120})
cola.Sales.plot()
lin_pred.plot(color='red')

#  -->Exponential Model
exp_model=smf.ols('log_Sales~t',data=train).fit()
exp_pred=exp_model.predict(test['t'])
exp_rmse=np.sqrt(np.mean((np.exp(exp_pred)-test['Sales'])**2))

cola.Sales.plot()
np.exp(exp_pred).plot(color='red')

#  -->Quadratic Model
qua_model=smf.ols('Sales~t+t_sq',data=train).fit()
qua_pred=qua_model.predict(test[['t','t_sq']])
qua_rmse=np.sqrt(np.mean((qua_pred-test['Sales'])**2))

cola.Sales.plot()
qua_pred.plot(color='red')

#  -->Linear Model additive seasonality
lin_add_model=smf.ols('Sales~t+Q1+Q2+Q3',data=train).fit() #three dummy variables used
lin_add_pred=lin_add_model.predict(test)
lin_add_rmse=np.sqrt(np.mean((lin_add_pred-test['Sales'])**2))

cola.Sales.plot()
lin_add_pred.plot(color='red')

#  -->Linear Model multiplicative seasonality
lin_mul_model=smf.ols('log_Sales~t+Q1+Q2+Q3',data=train).fit() #three dummy variables used
lin_mul_pred=lin_mul_model.predict(test)
lin_mul_rmse=np.sqrt(np.mean((np.exp(lin_mul_pred)-test['Sales'])**2))

cola.Sales.plot()
np.exp(lin_mul_pred).plot(color='red')

#  -->Quadratic Model additive seasonality
qua_add_model=smf.ols('Sales~t+t_sq+Q1+Q2+Q3',data=train).fit() #three dummy variables used
qua_add_pred=qua_add_model.predict(test)
qua_add_rmse=np.sqrt(np.mean((qua_add_pred-test['Sales'])**2))

cola.Sales.plot()
qua_add_pred.plot(color='red')

#  -->Quadratic Model multiplicative seasonality
qua_mul_model=smf.ols('log_Sales~t+t_sq+Q1+Q2+Q3',data=train).fit() #three dummy variables used
qua_mul_pred=qua_mul_model.predict(test)
qua_mul_rmse=np.sqrt(np.mean((np.exp(qua_mul_pred)-test['Sales'])**2))

cola.Sales.plot()
np.exp(qua_mul_pred).plot(color='red')

rmse_dict={'Linear':lin_rmse,
           'Exponential':exp_rmse,
           'Quadratic':qua_rmse,
           'Linear Additive':lin_add_rmse,
           'Linear Multiplicative':lin_mul_rmse,
           'Quadratic Additive':qua_add_rmse,
           'Quadratic Multiplicative':qua_mul_rmse}

min(rmse_dict.values())

# The quadratic additive model has the least rmse and 
# hence we go with Quadratic model with additive seasonality 


#####   Data Driven Techniques   #####
from statsmodels.graphics import tsaplots
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

tsaplots.plot_acf(cola.Sales,lags=20)
tsaplots.plot_pacf(cola.Sales,lags=20)
# as the data has trend and seasonality, Holt Winters method is the right one

#  -->Simple Exponential Smoothing
ses_model1=SimpleExpSmoothing(train.Sales).fit()
fcast1=ses_model1.forecast(10)

ses_model2=SimpleExpSmoothing(train.Sales).fit(smoothing_level=0.2)
fcast2=ses_model2.forecast(10)

ses_model3=SimpleExpSmoothing(train.Sales).fit(smoothing_level=0.6)
fcast3=ses_model3.forecast(10)

ses_model1.fittedvalues.plot()
fcast1.plot(color='blue',legend=True,label='alpha =%s'%ses_model1.model.params['smoothing_level'])
ses_model2.fittedvalues.plot()
fcast2.plot(color='red',legend=True,label='alpha = 0.2')
ses_model3.fittedvalues.plot()
fcast3.plot(color='yellow',legend=True,label='alpha = 0.6')
cola.Sales.plot(legend=True,label='Original')

def MAPE(pred,org):
    temp=np.abs(pred-org)*100/org
    return np.mean(temp)
ses_mape_dict={'ses_model1':MAPE(fcast1,test.Sales),'ses_model2':MAPE(fcast2,test.Sales),'ses_model3':MAPE(fcast3,test.Sales)}  

#  -->Holts Exponential Smoothing
holt_model1=Holt(train.Sales).fit(smoothing_level=0.8,smoothing_slope=0.2,optimized=False)
fcast1=holt_model1.forecast(10)

holt_model2=Holt(train.Sales,exponential=True).fit(smoothing_level=0.8,smoothing_slope=0.2,optimized=False)
fcast2=holt_model2.forecast(10)

holt_model3=Holt(train.Sales,damped=True).fit(smoothing_level=0.8,smoothing_slope=0.2)
fcast3=holt_model3.forecast(10)

holt_model1.fittedvalues.plot()
fcast1.plot(color='red',legend=True,label='Linear Trend')
holt_model2.fittedvalues.plot()
fcast2.plot(color='blue',legend=True,label='Exponential Trend')
holt_model3.fittedvalues.plot()
fcast3.plot(color='yellow',legend=True,label='Damped Trend')
cola.Sales.plot(label='Original',legend=True)

holt_mape_dict={'holt_linear':MAPE(fcast1,test.Sales),'holt_exponential':MAPE(fcast2,test.Sales),'holt_damped':MAPE(fcast3,test.Sales)}  

#  -->Holts Winters Smoothing
win_model1=ExponentialSmoothing(train.Sales,trend='additive',seasonal='additive',seasonal_periods=4).fit()
fcast1=win_model1.forecast(10)

win_model2=ExponentialSmoothing(train.Sales,trend='additive',seasonal='additive',seasonal_periods=4,damped=True).fit()
fcast2=win_model2.forecast(10)

win_model1.fittedvalues.plot()
fcast1.plot(color='blue',legend=True,label='Undamped')
win_model2.fittedvalues.plot()
fcast2.plot(color='yellow',legend=True,label='Damped')
cola.Sales.plot(legend=True,label='Original')

holt_winter_mape={'Undamped':MAPE(fcast1,test.Sales),'Damped':MAPE(fcast2,test.Sales)}

mape_dict={**ses_mape_dict,**holt_mape_dict,**holt_winter_mape}

# Holt's Winters undamped model has the least MAPE and hence can be used to forecast


