import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import preprocessing


salary_train=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/SalaryData_Train.csv")
salary_test=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/SalaryData_Test.csv")
salary_train.head()
salary_test.head()

salary=pd.concat([salary_train,salary_test])

#EDA
salary.isnull().sum()#NO NA values
salary.describe()
salary.columns

sns.countplot(x='Salary',data=salary)# more people with less than 50k salary

sns.boxplot(salary['age']) # outliers to the upper extreme
sns.boxplot(salary['educationno'])
sns.boxplot(salary['capitalgain'])# median 0
sns.boxplot(salary['capitalloss'])# median 0
sns.boxplot(salary['hoursperweek'])# outliers on both sides

sns.distplot(salary['age'],bins=15)#right skewed ; more people are young
sns.distplot(salary['capitalgain'],bins=15)# many with 0
sns.distplot(salary['capitalloss'],bins=15)# many with 0
sns.distplot(salary['hoursperweek'],bins=15)# many in mid range

sns.boxplot(x='Salary',y='age',data=salary)#median age higher for >50k salaried
sns.boxplot(x='Salary',y='capitalgain',data=salary)
sns.boxplot(x='Salary',y='capitalloss',data=salary)
sns.boxplot(x='Salary',y='hoursperweek',data=salary)# median higher for >50k salaried

sns.boxplot(x='Salary',y='capitalloss',data=salary.loc[salary['capitalloss']!=0]) # ignoring the 0 values,median loss higher for >50k salaried 
sns.boxplot(x='Salary',y='capitalgain',data=salary.loc[salary['capitalgain']!=0]) # ignoring the 0 values,median gain higher for >50k salaried 

sns.countplot(x='Salary',hue='workclass',data=salary) #private class more for both the groups
sns.countplot(x='Salary',hue='education',data=salary) #higher qualification for >50k group
sns.countplot(x='Salary',hue='maritalstatus',data=salary)#>50k mainly Married-civ-spouse;never married and divorced very less;<50k mainly never married,Married-civ-spouse and divorced
sns.countplot(x='Salary',hue='occupation',data=salary)
sns.countplot(x='Salary',hue='relationship',data=salary)# >50k mainly husbands, while <50k mainly not in family,husband,own child and unmarried 
sns.countplot(x='Salary',hue='race',data=salary)#non whites very less in >50k category
sns.countplot(x='Salary',hue='sex',data=salary)#females proportionately less for >50k
sns.countplot(x='Salary',hue='native',data=salary)#mostly US


train_x=salary_train.iloc[:,:-1]
train_y=salary_train.iloc[:,-1]
test_x=salary_test.iloc[:,:-1]
test_y=salary_test.iloc[:,-1]

salary_train.columns
cat_var=['workclass', 'education','maritalstatus','occupation', 'relationship', 'race', 'sex','native']
for i in cat_var:
    number=preprocessing.LabelEncoder()
    train_x[i]=number.fit_transform(train_x[i])
    test_x[i]=number.fit_transform(test_x[i])
    

model=SVC(kernel="rbf",)
model.fit(train_x,train_y)
pred=model.predict(train_x)
pd.crosstab(train_y,pred)
np.mean(train_y==pred)  # training accuracy=91.35%
pred=model.predict(test_x)
pd.crosstab(test_y,pred)
np.mean(test_y==pred)  #test accuracy=81.28%

model=SVC(kernel="linear",max_iter=500,probability=True)
model.fit(train_x,train_y)
pred=model.predict(train_x)
pd.crosstab(train_y,pred)
np.mean(train_y==pred)  # training accuracy=65.76%
pred=model.predict(test_x)
pd.crosstab(test_y,pred)
np.mean(test_y==pred)  #test accuracy=64.15

#############################################################################

fire=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\forestfires.csv")
fire.head()
fire["size_category"].unique()

fire.drop(["month","day"],axis=1,inplace=True)

fire.loc[fire.size_category=="small","size_category"]=0
fire.loc[fire.size_category=="large","size_category"]=1

x=fire.iloc[:,:-1]
y=fire.iloc[:,-1]

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2)

linear_model=SVC(kernel="linear")
linear_model.fit(train_x,train_y)
linear_pred=linear_model.predict(train_x)
pd.crosstab(train_y,linear_pred)
np.mean(train_y==linear_pred)  # training accuracy=100%
linear_pred=linear_model.predict(test_x)
pd.crosstab(test_y,linear_pred)
np.mean(test_y==linear_pred)  # test accuracy=99.03%

poly_model=SVC(kernel="poly")
poly_model.fit(train_x,train_y)
poly_pred=poly_model.predict(train_x)
pd.crosstab(train_y,poly_pred)
np.mean(train_y==poly_pred)  # training accuracy=100%
poly_pred=poly_model.predict(test_x)
pd.crosstab(test_y,poly_pred)
np.mean(test_y==poly_pred)  # test accuracy=97.11%

rbf_model=SVC(kernel="rbf")
rbf_model.fit(train_x,train_y)
rbf_pred=rbf_model.predict(train_x)
pd.crosstab(train_y,rbf_pred)
np.mean(train_y==rbf_pred) # training accuracy=99.75%
rbf_pred=rbf_model.predict(test_x)
pd.crosstab(test_y,rbf_pred)
np.mean(test_y==rbf_pred) #test accuracy=81.73%

