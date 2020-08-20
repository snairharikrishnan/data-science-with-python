import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

Amtrak=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\Amtrak.csv")

Amtrak.rename(columns={'Ridership ':'Ridership'},inplace=True)
Amtrak.describe()
Amtrak.plot()       # seems to be quadratic trend and additive seasonality
Amtrak.columns
mon=Amtrak['Month'][0]
mon[0:3]
Amtrak["Months"]=0

for i in range(159):
    mon=Amtrak["Month"][i][0:3]
    Amtrak["Months"][i]=mon

list(Amtrak["Months"].unique())    
dummy_var=pd.get_dummies(Amtrak['Months'])
Amtrak1=pd.concat([Amtrak,dummy_var],axis=1)

Amtrak1["t"]=np.arange(1,160)
Amtrak1["t_sq"]=Amtrak1["t"]*Amtrak1["t"]
Amtrak1["log_Ridership"]=np.log(Amtrak1['Ridership'])
Train=Amtrak1.head(147)
Test=Amtrak1.tail(12)

# Linear Model
import statsmodels.formula.api as smf
from sklearn.metrics import explained_variance_score

lin_model=smf.ols('Ridership~t',data=Train).fit()
lin_pred=lin_model.predict(Test['t'])
explained_variance_score(Test['Ridership'],lin_pred)
lin_rmse=np.sqrt(np.mean((Test['Ridership']-lin_pred)**2))
x=np.arange(12)
plt.plot(x,lin_pred,'r',x,Test['Ridership'],'b')

# Exponential Model
exp_model=smf.ols('log_Ridership~t',data=Train).fit()
exp_pred=exp_model.predict(Test['t'])
explained_variance_score(Test['Ridership'],np.exp(exp_pred))
exp_rmse=np.sqrt(np.mean((Test['Ridership']-np.exp(exp_pred))**2))
plt.plot(x,np.exp(exp_pred),'r',x,Test['Ridership'],'b')

# Quardatic Model
qua_model=smf.ols('Ridership~t+t_sq',data=Train).fit()
qua_pred=qua_model.predict(Test[['t','t_sq']])
explained_variance_score(Test['Ridership'],qua_pred)
qua_rmse=np.sqrt(np.mean((Test['Ridership']-qua_pred)**2))
plt.plot(x,qua_pred,'r',x,Test['Ridership'],'b')

#Linear Model Additive Seasonality
lin_add_model=smf.ols('Ridership~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
lin_add_pred=lin_add_model.predict(Test[['t','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']])
explained_variance_score(Test['Ridership'],lin_add_pred)
lin_add_rmse=np.sqrt(np.mean((Test['Ridership']-lin_add_pred)**2))
plt.plot(x,lin_add_pred,'r',x,Test['Ridership'],'b')

#Linear Model Multiplicative Seasonality
lin_mul_model=smf.ols('log_Ridership~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
lin_mul_pred=lin_mul_model.predict(Test[['t','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']])
explained_variance_score(Test['Ridership'],np.exp(lin_mul_pred))
lin_mul_rmse=np.sqrt(np.mean((Test['Ridership']-np.exp(lin_mul_pred))**2))
plt.plot(x,np.exp(lin_mul_pred),'r',x,Test['Ridership'],'b')

# Quadratic Model Additive Seasonality
qua_add_model=smf.ols('Ridership~t+t_sq+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
qua_add_pred=qua_add_model.predict(Test[['t','t_sq','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']])
explained_variance_score(Test['Ridership'],qua_add_pred)
qua_add_rmse=np.sqrt(np.mean((Test['Ridership']-qua_add_pred)**2))
plt.plot(x,qua_add_pred,'r',x,Test['Ridership'],'b')
a=np.arange(147,159)
b=np.arange(159)
plt.plot(a,qua_add_pred,'r',b,Amtrak1['Ridership'],'b')

# Quadratic Model Multiplicative Seasonality
qua_mul_model=smf.ols('log_Ridership~t+t_sq+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
qua_mul_pred=qua_mul_model.predict(Test[['t','t_sq','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']])
explained_variance_score(Test['Ridership'],np.exp(qua_mul_pred))
qua_mul_rmse=np.sqrt(np.mean((Test['Ridership']-np.exp(qua_mul_pred))**2))
plt.plot(x,np.exp(qua_mul_pred),'r',x,Test['Ridership'],'b')


data={'MODEL':['lin_model','exp_model','qua_model','lin_add_model','lin_mul_model','qua_add_model','qua_mul_model'],
      'RMSE':[lin_rmse,exp_rmse,qua_rmse,lin_add_rmse,lin_mul_rmse,qua_add_rmse,qua_mul_rmse]}
table=pd.DataFrame(data)







