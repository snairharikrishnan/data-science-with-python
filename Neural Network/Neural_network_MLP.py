import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wbcd=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\wbcd.csv")
wbcd.head()
wbcd.columns
plt.hist(wbcd["diagnosis"])
wbcd.drop(columns=["id"],axis=1,inplace=True)

wbcd.loc[wbcd.diagnosis=='B',"diagnosis"]=0
wbcd.loc[wbcd.diagnosis=='M',"diagnosis"]=1
wbcd.diagnosis.values

colnames=wbcd.columns
predictors=colnames[1:]
target=colnames[0]

from sklearn.model_selection import train_test_split
train,test=train_test_split(wbcd,test_size=0.2)

from sklearn.preprocessing import StandardScaler 
scaler=StandardScaler()
scaler.fit(train[predictors])

train[predictors]=scaler.transform(train[predictors])
test[predictors]=scaler.transform(test[predictors])

from sklearn.neural_network import MLPClassifier
help(MLPClassifier)
mlp=MLPClassifier(hidden_layer_sizes=(20,20),activation='relu')
mlp.fit(train[predictors],train[target])

pred_train=mlp.predict(train[predictors])
pd.crosstab(train[target],pred_train)
train_accuracy=np.mean(train[target]==pred_train) #99.56

pred_test=mlp.predict(test[predictors])
pd.crosstab(test[target],pred_test)
teat_accuracy=np.mean(test[target]==pred_test)  #96.49
