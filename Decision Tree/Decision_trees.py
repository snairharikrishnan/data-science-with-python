import pandas as pd
import numpy as np

iris=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/iris.csv")
iris.head()
iris.Species.value_counts()

from sklearn.model_selection import train_test_split
train,test = train_test_split(iris,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier as DTC
model=DTC(criterion="entropy")
model.fit(train.iloc[:,:-1],train.iloc[:,-1])

pred=model.predict(train.iloc[:,:-1])
pd.crosstab(pred,train.iloc[:,-1])
train_acc=np.mean(pred==train.iloc[:,-1]) #1.0

pred_test=model.predict(test.iloc[:,:-1])
pd.crosstab(pred_test,test.iloc[:,-1])
test_acc=np.mean(pred_test==test.iloc[:,-1]) #0.86