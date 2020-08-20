import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics import tsaplots
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing



Amtrak=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\Amtrak.csv")
Amtrak.rename(columns={'Ridership ':'Ridership'},inplace=True)
Amtrak.index = pd.to_datetime(Amtrak.Month,format="%b-%y")

Amtrak.Ridership.plot(figsize=(15,7),label="Original")
for i in range(2,24,6):
    Amtrak.Ridership.rolling(i).mean().plot(figsize=(15,7),label=i)
plt.legend()

plt.rcParams.update({'figure.figsize':(8,5),'figure.dpi':100})
decompose_ts_add = seasonal_decompose(Amtrak.Ridership,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(Amtrak.Ridership,model="multiplicative")
decompose_ts_mul.plot()

tsaplots.plot_acf(Amtrak.Ridership,lags=10)
tsaplots.plot_pacf(Amtrak.Ridership)

Train = Amtrak.head(133)
Test = Amtrak.tail(12)

#Simple Exponential Smoothing
#Few data points, Irregular data, No seasonality or trend.
ses_model=SimpleExpSmoothing(Train['Ridership'])
fit1=ses_model.fit()
pred1=fit1.forecast(12).rename(r'$\alpha=%s$'%fit1.model.params['smoothing_level'])

fit2=ses_model.fit(smoothing_level=0.2)
pred2=fit2.forecast(12).rename(r'$\alpha=0.2$')

fit3=ses_model.fit(smoothing_level=0.6)
pred3=fit3.forecast(12).rename(r'$\alpha=0.6$')


pred1.plot(color='blue', legend=True)
fit1.fittedvalues.plot(color='blue')
pred2.plot(color='red', legend=True)
fit2.fittedvalues.plot(color='red')
pred3.plot(color='green', legend=True)
fit3.fittedvalues.plot(color='green')
Amtrak.Ridership.plot(color='black', legend=True)


# Holt’s Linear Smoothing , Trend in data, No seasonality.
fit1 = Holt(Train['Ridership']).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fcast1 = fit1.forecast(12).rename("Holt's linear trend")

fit2 = Holt(Train['Ridership'], exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
fcast2 = fit2.forecast(12).rename("Exponential trend")

fit3 = Holt(Train['Ridership'], damped=True).fit(smoothing_level=0.8, smoothing_slope=0.2)
fcast3 = fit3.forecast(12).rename("Additive damped trend")


fit1.fittedvalues.plot(color='blue')
fcast1.plot(color='blue', legend=True)
fit2.fittedvalues.plot(color='red')
fcast2.plot(color='red',legend=True)
fit3.fittedvalues.plot(color='green')
fcast3.plot(color='green', legend=True)
Amtrak.Ridership.plot(color='black', legend=True)


def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

x=pd.DataFrame(dict(pred1).values())
y=pd.DataFrame(dict(Test.Ridership).values())
MAPE(x,y)


#Holt Winter's
fit1=ExponentialSmoothing(Train['Ridership'],trend='add',seasonal='add',seasonal_periods=12).fit()
fcast1=fit1.forecast(12)
fit2=ExponentialSmoothing(Train['Ridership'],trend='add',seasonal='add',seasonal_periods=12,damped=True).fit()
fcast2=fit2.forecast(12)

sse1 = np.sqrt(np.mean(np.square(Test.Ridership.values - fcast1.values)))
sse2 = np.sqrt(np.mean(np.square(Test.Ridership.values - fcast2.values)))

fit1.fittedvalues.plot(style='--', color='red')
fit2.fittedvalues.plot(style='--', color='green')
fcast1.plot(style='--', marker='o', color='red', legend=True)
fcast2.plot(style='--', marker='o', color='green', legend=True)
Amtrak.Ridership.plot(color='gray', legend=True)






