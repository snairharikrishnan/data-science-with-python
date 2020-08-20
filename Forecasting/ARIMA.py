#link -> https://www.machinelearningplus.com/time-series/arima-model-time-series-forecasting-python/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
from statsmodels.tsa.arima_model import ARIMA 

Amtrak=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\Amtrak.csv")
Amtrak.rename(columns={'Ridership ':'Ridership'},inplace=True)

tsaplots.plot_acf(Amtrak.Ridership,lags=12)
tsaplots.plot_pacf(Amtrak.Ridership,lags=12)

model1=ARIMA(Amtrak.Ridership,order=(1,1,3)).fit(disp=0)  #order=(p,d,q)
model2=ARIMA(Amtrak.Ridership,order=(1,1,4)).fit(disp=0)

model1.aic
model2.aic      # choose model with least aic


# To find d
from statsmodels.tsa.stattools import adfuller
#H0 = time series is non-stationary
result = adfuller(Amtrak.Ridership.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])  #p>0.05 accept H0 Hence differencing required

plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

#Original Series
fig, axes = plt.subplots(3, 2, sharex=False)
axes[0, 0].plot(Amtrak.Ridership); axes[0, 0].set_title('Original Series')
tsaplots.plot_acf(Amtrak.Ridership, ax=axes[0, 1])

# 1st Differencing
axes[1, 0].plot(Amtrak.Ridership.diff()); axes[1, 0].set_title('1st Order Differencing')
tsaplots.plot_acf(Amtrak.Ridership.diff().dropna(), ax=axes[1, 1])

# 2nd Differencing
axes[2, 0].plot(Amtrak.Ridership.diff().diff()); axes[2, 0].set_title('2nd Order Differencing')
tsaplots.plot_acf(Amtrak.Ridership.diff().diff().dropna(), ax=axes[2, 1])

#For the above series, the time series reaches stationarity with one order of differencing. 
#But on looking at the autocorrelation plot for the 2nd differencing the lag goes into the far
#negative zone fairly quick, which indicates, the series might have been over differenced.

# plot becomes stationary with one difference 
# Hence d=1


#To find p --> PACF plot of stationary series
# PACF plot of 1st differenced series
plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

fig, axes = plt.subplots(1, 2, sharex=False)
axes[0].plot(Amtrak.Ridership.diff()); axes[0].set_title('1st Differencing')
axes[1].set(ylim=(0,1))
tsaplots.plot_pacf(Amtrak.Ridership.diff().dropna(), ax=axes[1])
# see how may points cross significant level in the starting, here we take 1

p=1
d=1
pdq=[]
aic=[]

for q in range(10):
    try:
        model=ARIMA(Amtrak.Ridership,order=(p,d,q)).fit(disp=0)
        aic.append(model.aic)
        pdq.append((p,d,q))
    except:
        pass

dict(zip(pdq,aic))
a=np.arange(10)
plt.plot(a,aic)

#least aic for 1,1,8

model=ARIMA(Amtrak.Ridership,order=(1,1,6)).fit(disp=0)
print(model.summary())


# Plot residual errors
residuals = pd.DataFrame(model.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
fig

model.plot_predict(dynamic=False)


#Seasonal data requires SARIMA
fig, axes = plt.subplots(2, 1, figsize=(10,5), dpi=100, sharex=True)

# Usual Differencing
axes[0].plot(Amtrak.Ridership, label='Original Series')
axes[0].plot(Amtrak.Ridership.diff(1), label='Usual Differencing')
axes[0].set_title('Usual Differencing')
axes[0].legend(loc='upper left', fontsize=10)

# Seasinal Dei
axes[1].plot(Amtrak.Ridership, label='Original Series')
axes[1].plot(Amtrak.Ridership.diff(12), label='Seasonal Differencing', color='green')
axes[1].set_title('Seasonal Differencing')
plt.legend(loc='upper left', fontsize=10)
plt.suptitle('a10 - Drug Sales', fontsize=16)
fig

import pmdarima as pm  # --> issue










