import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

housing=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\housing_XGBRegressor.csv")
housing.isnull().sum()

X=housing.iloc[:,:-1]
Y=housing.iloc[:,-1]

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(X,Y,test_size=0.25,random_state=0)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

lin_model=LinearRegression()
lin_model.fit(train_x,train_y)
pred=lin_model.predict(train_x)
lin_model.score(train_x,train_y)  #R squared = 76.96%
np.sqrt(np.mean((pred-train_y)**2)) #RMSE = 4.43

pred=lin_model.predict(test_x)
lin_model.score(test_x,test_y)  #R squared = 63.54%
np.sqrt(np.mean((pred-test_y)**2)) #RMSE = 5.45


ridge_model=Ridge(alpha=0.5,normalize=True)
ridge_model.fit(train_x,train_y)
pred=ridge_model.predict(train_x)
ridge_model.score(train_x,train_y)  #R squared = 72.2%
np.sqrt(np.mean((pred-train_y)**2)) #RMSE = 4.86

#Hyperparameter tuning
R_sq=[]
train_rmse=[]
test_rmse=[]
alphas=np.arange(0,5,0.001)

for i in alphas:
    ridge_model=Ridge(alpha=i,normalize=True)
    ridge_model.fit(train_x,train_y)
    pred_train=ridge_model.predict(train_x)
    R_sq.append(ridge_model.score(train_x,train_y))
    train_rmse.append(np.sqrt(np.mean((pred_train-train_y)**2)) )
    pred_test=ridge_model.predict(test_x)
    test_rmse.append(np.sqrt(np.mean((pred_test-test_y)**2)))


plt.scatter(x=alphas,y=R_sq)
plt.scatter(x=alphas,y=train_rmse)
plt.scatter(x=alphas,y=test_rmse)


ridge_model=Ridge(alpha=0.001,normalize=True)
ridge_model.fit(train_x,train_y)
pred=ridge_model.predict(test_x)
ridge_model.score(test_x,test_y)  
np.sqrt(np.mean((pred-test_y)**2)) 


#Lasso

R_sq=[]
train_rmse=[]
test_rmse=[]
alphas=np.arange(0,5,0.001)

for i in alphas:
    lasso_model=Lasso(alpha=i,normalize=True)
    lasso_model.fit(train_x,train_y)
    pred_train=lasso_model.predict(train_x)
    R_sq.append(lasso_model.score(train_x,train_y))
    train_rmse.append(np.sqrt(np.mean((pred_train-train_y)**2)) )
    pred_test=lasso_model.predict(test_x)
    test_rmse.append(np.sqrt(np.mean((pred_test-test_y)**2)))

plt.scatter(x=alphas,y=R_sq)
plt.scatter(x=alphas,y=train_rmse)
plt.scatter(x=alphas,y=test_rmse)

lasso_model=Lasso(alpha=0.001,normalize=True)
lasso_model.fit(train_x,train_y)
pred_train=lasso_model.predict(train_x)
lasso_model.score(train_x,train_y)
np.sqrt(np.mean((pred_train-train_y)**2)) 
pred_test=lasso_model.predict(test_x)
np.sqrt(np.mean((pred_test-test_y)**2))



