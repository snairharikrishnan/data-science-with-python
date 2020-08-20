import pandas as pd
import numpy as np

diabetes=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/Diabetes_RF.csv")
diabetes.head()
diabetes.columns
diabetes.describe()

colnames=list(diabetes.columns)
target=diabetes[colnames[-1]]
predictor=diabetes[colnames[:-1]]

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(criterion="entropy",n_estimators=15,n_jobs=3,oob_score=True)

rf.fit(predictor,target)
rf.estimators_
rf.classes_
rf.n_classes_
rf.n_features_
rf.n_outputs_
rf.oob_score_

pred=rf.predict(predictor)

diabetes["Prediction"]=pred

pd.crosstab(diabetes[" Class variable"],diabetes["Prediction"])

print("Accuracy = ",(499+265)/(499+265+4))  #99.47%

from sklearn.metrics import confusion_matrix
confusion_matrix(diabetes[' Class variable'],diabetes['Prediction']) # Confusion matrix

################################################################################

salary_train=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/SalaryData_Train.csv")
salary_test=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/SalaryData_Test.csv")

salary_train.head()
salary_test.head()
 
colnames = salary_train.columns

trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]

categorical_var=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
from sklearn import preprocessing
for i in categorical_var:
    number=preprocessing.LabelEncoder()
    trainX[i]=number.fit_transform(trainX[i])

trainX.head()

rfsalary = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=15,criterion="entropy")
rfsalary.fit(trainX,trainY) 
 
pred=rfsalary.predict(trainX)

table=confusion_matrix(trainY,pred)
accuracy=(table[0,0]+table[1,1])/(table[0,0]+table[0,1]+table[1,0]+table[1,1]) #97%


testX = salary_test[colnames[0:13]]
testY = salary_test[colnames[13]]

for i in categorical_var:
    number=preprocessing.LabelEncoder()
    testX[i]=number.fit_transform(testX[i])

testX.head()
 
pred=rfsalary.predict(testX)

table=confusion_matrix(testY,pred)
accuracy=(table[0,0]+table[1,1])/(table[0,0]+table[0,1]+table[1,0]+table[1,1]) #84.09%



