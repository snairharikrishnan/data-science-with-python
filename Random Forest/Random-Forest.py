import pandas as pd
import numpy as np
import seaborn as sns

fraud=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/Fraud_check.csv")
fraud.head()
fraud.describe()
fraud.columns

#EDA
fraud.isnull().sum()# no NA values
sns.distplot(fraud['Taxable.Income'],bins=20) # almost normal
sns.boxplot(fraud['Taxable.Income'])# no outliers
sns.distplot(fraud['City.Population'],bins=15)# normal distribution
sns.distplot(fraud['Work.Experience'],bins=15)# a bit left skewed

sns.scatterplot(x='Taxable.Income',y='City.Population',data=fraud)
sns.scatterplot(x='Taxable.Income',y='Work.Experience',data=fraud)
fraud.corr()#all weakly correlated

sns.catplot(x='Undergrad',y='Taxable.Income',data=fraud,kind='box') #almost same
sns.catplot(x='Marital.Status',y='Taxable.Income',data=fraud,kind='box')#median income less for married
sns.catplot(x='Urban',y='Taxable.Income',data=fraud,kind='box')#median income a bit more for Urban people
sns.catplot(x='Urban',y='Taxable.Income',hue='Marital.Status',data=fraud)
sns.catplot(x='Urban',y='Taxable.Income',hue='Undergrad',data=fraud)
sns.catplot(x='Marital.Status',y='Taxable.Income',hue='Undergrad',data=fraud)
sns.catplot(x='Marital.Status',y='Taxable.Income',hue='Urban',data=fraud)
sns.catplot(x='Undergrad',y='Taxable.Income',hue='Marital.Status',data=fraud)
sns.catplot(x='Undergrad',y='Taxable.Income',hue='Urban',data=fraud)# no clear distinction


categorical_var=["Undergrad","Marital.Status","Urban"]
from sklearn import preprocessing
for i in categorical_var:
    number=preprocessing.LabelEncoder()
    fraud[i]=number.fit_transform(fraud[i])
    
fraud.head()

income_range=pd.cut(fraud["Taxable.Income"],bins=[0,30000,100000],labels=["Risky","Good"])
fraud["Taxable.Income"]=income_range
fraud.head()

colnames=list(fraud.columns)
target=colnames.pop(2)
predictors=colnames

from sklearn.model_selection import train_test_split
train,test=train_test_split(fraud,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(criterion="gini",n_estimators=17, oob_score=True)
model.fit(train[predictors],train[target])

pred=model.predict(train[predictors])
pd.crosstab(train[target],pred)
train_accuracy=np.mean(train[target]==pred) # 99.37%

pred=model.predict(test[predictors])
pd.crosstab(test[target],pred)
test_accuracy=np.mean(test[target]==pred) # 75%  

############################################################

company=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/Company_Data.csv")
company.head()
company.describe()

categorical_var=["ShelveLoc","Urban","US"]
for i in categorical_var:
    number=preprocessing.LabelEncoder()
    company[i]=number.fit_transform(company[i])

company.head()

sales_range=pd.cut(company["Sales"],bins=3,labels=["Low","Medium","High"])
company["Sales"]=sales_range
company.head()

colnames=list(company.columns)
target=colnames[0]
predictors=colnames[1:]

train,test=train_test_split(company,test_size=0.2)

model=RandomForestClassifier(criterion="gini",n_estimators=13,oob_score=True)
model.fit(train[predictors],train[target])

pred=model.predict(train[predictors])
pd.crosstab(train[target],pred)
train_acc=np.mean(train[target]==pred) # 99.68%

pred=model.predict(test[predictors])
pd.crosstab(test[target],pred)
test_acc=np.mean(test[target]==pred) # 80%

#########################################################################

iris=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Project\\bank_clean.csv")
iris.head()
iris=iris.iloc[:500,6:]
colnames=list(iris.columns)
predictors=colnames[:-1]
target=colnames[1]

train,test=train_test_split(iris,test_size=0.2)

model=RandomForestClassifier(criterion="entropy",n_estimators=18,oob_score=True)
model.fit(train[predictors],train[target])

pred=model.predict(train[predictors])
pd.crosstab(train[target],pred)
train_acc=np.mean(train[target]==pred) # 100%

pred=model.predict(test[predictors])
pd.crosstab(test[target],pred)
test_acc=np.mean(test[target]==pred) # 90%





