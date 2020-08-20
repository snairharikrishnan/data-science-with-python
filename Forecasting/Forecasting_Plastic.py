import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import statsmodels.formula.api as smf

plastic=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\PlasticSales.csv")

plt.rcParams.update({'figure.figsize':(7,5),'figure.dpi':120})
plastic['Sales'].plot() #upward linear trend with additive seasonaity
plastic.rename(columns={'Month':'Month_Year'},inplace=True)

plastic['Month_Year'][0][0:3]
int(plastic['Month_Year'][0][4:])+1900

plastic['Month']=0
plastic['Year']=0
for i in range(len(plastic)):
    plastic['Month'][i]=plastic['Month_Year'][i][0:3]
    plastic['Year'][i]=int(plastic['Month_Year'][i][4:])+1900

sns.distplot(plastic['Sales']) #normal distribution
plastic.Sales.rolling(12).mean().plot()
sns.boxplot(x='Month',y='Sales',data=plastic) #sales peak in Aug,Sep,Oct,least in feb
sns.boxplot(x='Year',y='Sales',data=plastic) #sales increases year on

plastic['t']=np.arange(1,61)
plastic['t_sq']=plastic['t']*plastic['t']
plastic['log_Sales']=np.log(plastic['Sales'])
dummy=pd.get_dummies(plastic['Month'])   #dummy variables for the months
plastic=pd.concat([plastic,dummy],axis=1)

train=plastic.iloc[:-15,:]
test=plastic.iloc[-15:,:]

#Model Based Techniques

#Linear model
lin_model=smf.ols('Sales~t',data=train).fit()
lin_pred=lin_model.predict(test['t'])
lin_rmse=np.sqrt(np.mean((lin_pred-test['Sales'])**2))

plastic.Sales.plot()
lin_pred.plot(color='red')

#Exponential model
exp_model=smf.ols('log_Sales~t',data=train).fit()
exp_pred=exp_model.predict(test['t'])
exp_rmse=np.sqrt(np.mean((np.exp(exp_pred)-test['Sales'])**2))

plastic.Sales.plot()
np.exp(exp_pred).plot(color='red')

#Quadratic Model
qua_model=smf.ols('Sales~t+t_sq',data=train).fit()
qua_pred=qua_model.predict(test[['t','t_sq']])
qua_rmse=np.sqrt(np.mean((qua_pred-test['Sales'])**2))

plastic.Sales.plot()
qua_pred.plot(color='red')

#Linear Model additive seasonality
lin_add_model=smf.ols('Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit() 
lin_add_pred=lin_add_model.predict(test)
lin_add_rmse=np.sqrt(np.mean((lin_add_pred-test['Sales'])**2))

plastic.Sales.plot()
lin_add_pred.plot(color='red')

#Linear Model multiplicative seasonality
lin_mul_model=smf.ols('log_Sales~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
lin_mul_pred=lin_mul_model.predict(test)
lin_mul_rmse=np.sqrt(np.mean((np.exp(lin_mul_pred)-test['Sales'])**2))

plastic.Sales.plot()
np.exp(lin_mul_pred).plot(color='red')

#Quadratic Model additive seasonality
qua_add_model=smf.ols('Sales~t+t_sq+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
qua_add_pred=qua_add_model.predict(test)
qua_add_rmse=np.sqrt(np.mean((qua_add_pred-test['Sales'])**2))

plastic.Sales.plot()
qua_add_pred.plot(color='red')

#Quadratic Model multiplicative seasonality
qua_mul_model=smf.ols('log_Sales~t+t_sq+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=train).fit()
qua_mul_pred=qua_mul_model.predict(test)
qua_mul_rmse=np.sqrt(np.mean((np.exp(qua_mul_pred)-test['Sales'])**2))

plastic.Sales.plot()
np.exp(qua_mul_pred).plot(color='red')

rmse_dict={'Linear':lin_rmse,
           'Exponential':exp_rmse,
           'Quadratic':qua_rmse,
           'Linear Additive':lin_add_rmse,
           'Linear Multiplicative':lin_mul_rmse,
           'Quadratic Additive':qua_add_rmse,
           'Quadratic Multiplicative':qua_mul_rmse}

min(rmse_dict.values())
#Linear model with additive seasonality fits best as it has least rmse

#Data riven Techniques
from statsmodels.graphics import tsaplots
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

tsaplots.plot_acf(plastic.Sales,lags=20)
tsaplots.plot_pacf(plastic.Sales,lags=20)
#Holt Winters technique to be used as seasonality is present


#Simple Exponential Smoothing
ses_model1=SimpleExpSmoothing(train.Sales).fit()
fcast1=ses_model1.forecast(15)

ses_model2=SimpleExpSmoothing(train.Sales).fit(smoothing_level=0.8)
fcast2=ses_model2.forecast(15)

ses_model3=SimpleExpSmoothing(train.Sales).fit(smoothing_level=0.6)
fcast3=ses_model3.forecast(15)

ses_model1.fittedvalues.plot()
fcast1.plot(color='blue',legend=True,label='alpha =%s'%ses_model1.model.params['smoothing_level'])
ses_model2.fittedvalues.plot()
fcast2.plot(color='red',legend=True,label='alpha = 0.8')
ses_model3.fittedvalues.plot()
fcast3.plot(color='yellow',legend=True,label='alpha = 0.6')
plastic.Sales.plot(legend=True,label='Original')

def MAPE(pred,org):
    temp=np.abs(pred-org)*100/org
    return np.mean(temp)
ses_mape_dict={'ses_model1':MAPE(fcast1,test.Sales),'ses_model2':MAPE(fcast2,test.Sales),'ses_model3':MAPE(fcast3,test.Sales)}  

#Holts Exponential Smoothing
holt_model1=Holt(train.Sales).fit(smoothing_level=0.8,smoothing_slope=0.2,optimized=False)
fcast1=holt_model1.forecast(15)

holt_model2=Holt(train.Sales,exponential=True).fit(smoothing_level=0.8,smoothing_slope=0.2,optimized=False)
fcast2=holt_model2.forecast(15)

holt_model3=Holt(train.Sales,damped=True).fit(smoothing_level=0.8,smoothing_slope=0.2)
fcast3=holt_model3.forecast(15)

holt_model1.fittedvalues.plot()
fcast1.plot(color='red',legend=True,label='Linear Trend')
holt_model2.fittedvalues.plot()
fcast2.plot(color='blue',legend=True,label='Exponential Trend')
holt_model3.fittedvalues.plot()
fcast3.plot(color='yellow',legend=True,label='Damped Trend')
plastic.Sales.plot(label='Original',legend=True)

holt_mape_dict={'holt_linear':MAPE(fcast1,test.Sales),'holt_exponential':MAPE(fcast2,test.Sales),'holt_damped':MAPE(fcast3,test.Sales)}  

#Holts Winters Smoothing
win_model1=ExponentialSmoothing(train.Sales,trend='additive',seasonal='additive',seasonal_periods=12).fit()
fcast1=win_model1.forecast(15)

win_model2=ExponentialSmoothing(train.Sales,trend='additive',seasonal='additive',seasonal_periods=12,damped=True).fit()
fcast2=win_model2.forecast(15)

win_model1.fittedvalues.plot()
fcast1.plot(color='blue',legend=True,label='Undamped')
win_model2.fittedvalues.plot()
fcast2.plot(color='yellow',legend=True,label='Damped')
plastic.Sales.plot(legend=True,label='Original')

holt_winter_mape={'Undamped':MAPE(fcast1,test.Sales),'Damped':MAPE(fcast2,test.Sales)}

mape_dict={**ses_mape_dict,**holt_mape_dict,**holt_winter_mape}

#damped holt winters has least MAPE and hence can be used for prediction

