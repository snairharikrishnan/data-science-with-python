import pandas as pd
import numpy as np

diabetes=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/Diabetes_RF.csv")
diabetes.head()
diabetes.columns

output_var=pd.get_dummies(diabetes.iloc[:,-1],drop_first=True)
diabetes[" Class variable"]=output_var
diabetes.head()

x=diabetes.iloc[:,:-1]
y=diabetes.iloc[:,-1]

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2)

#Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#train_x=sc.fit_transform(train_x)
#test_x=sc.transform(train_x)

import lightgbm as lgb
d_train=lgb.Dataset(train_x,label=train_y)

params={}
params['learning_rate']=0.01
params['boosting_type']='gbdt'
params['objective']='binary'
params['metric']='binary_logloss'
params['sub_feature']=0.5
params['num_leaves']=10
params['min_data']=50
params['max_depth']=10

clf=lgb.train(params,d_train,100)
pred=clf.predict(train_x)

len(train_x)

for i in range(0,614):
    if pred[i]>=0.5:
        pred[i]=1
    else:
        pred[i]=0

train_y.dtype
pred.dtype
pred=pred.astype(int)

pd.crosstab(pred,train_y)
accuracy=np.mean(pred==train_y)  #77.85%


pred=clf.predict(test_x)

len(test_x)

for i in range(0,154):
    if pred[i]>=0.5:
        pred[i]=1
    else:
        pred[i]=0

test_y.dtype
pred.dtype
pred=pred.astype(int)

pd.crosstab(pred,test_y)
accuracy=np.mean(pred==test_y)  #77.27%











