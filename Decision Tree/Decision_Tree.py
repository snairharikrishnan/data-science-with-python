import pandas as pd
import numpy as np
import seaborn as sns

company=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/Company_Data.csv")
company.head()
company.columns

# EDA
company.isnull().sum()  # No NA values
company.iloc[:,:4].describe()
company.iloc[:,4:].describe()

company.iloc[:,6].value_counts()
company.iloc[:,9].value_counts()
company.iloc[:,10].value_counts()    

sns.distplot(company['Sales'],bins=15) # the output variable is seen to be normally distributed
company.plot.hist(subplots=True, layout=(5,2), figsize=(10, 10), bins=30)
sns.boxplot(company['Sales']) # two outliers to the upper extreme
sns.distplot(company['CompPrice'],bins=15)# normal distribution
sns.distplot(company['Income'],bins=15)# almost normal
sns.distplot(company['Advertising'],bins=15)#right skewed, no advertising budget for most
sns.distplot(company['Population'],bins=15)#normal
sns.distplot(company['Price'],bins=15)#normal
sns.distplot(company['Age'],bins=15)#normal
sns.distplot(company['Education'],bins=15)#normal

sns.scatterplot(x='Sales',y='CompPrice',data=company)
sns.scatterplot(x='Sales',y='Income',data=company)
sns.scatterplot(x='Sales',y='Advertising',data=company)
sns.scatterplot(x='Sales',y='Population',data=company)
sns.scatterplot(x='Sales',y='Price',data=company) #negative correlation,sales increases as price decreases
sns.scatterplot(x='Sales',y='Age',data=company)
sns.scatterplot(x='Sales',y='Education',data=company)

sns.pairplot(company)
x=company.corr()# all are weakly correlated
sns.catplot(x='Sales',y='ShelveLoc',data=company) #more sales for good ShelveLoc,least for bad ShelveLoc
sns.catplot(x='Sales',y='Urban',data=company)
sns.catplot(x='Sales',y='US',data=company)

ShelveLoc=[]
for i in company["ShelveLoc"]:
    if i=='Bad':
        ShelveLoc.append(0)
    elif i=='Medium':
        ShelveLoc.append(1)
    else:
        ShelveLoc.append(2)
        
company["ShelveLoc_num"]=ShelveLoc
company.iloc[:,[6,11]]
pd.crosstab(company.iloc[:,6],company.iloc[:,11])

company=company.drop(columns=["ShelveLoc"])
company.columns
company.head()

categorical_var = ["Urban","US"]
dummy_data = pd.get_dummies(data= company,columns=categorical_var,drop_first=True)

sales_range=pd.cut(dummy_data["Sales"],4,labels=[0,1,2,3]) # dividing into 4 equal ranges
dummy_data["Sales_range"]=sales_range
dummy_data.head()
dummy_data.iloc[:,[0,-1]]

#np.where(dummy_data["Sales"]<4)
#dummy_data.iloc[198,-1]
dummy_data=dummy_data.iloc[:,1:]
dummy_data.head()

colnames=list(dummy_data.columns)
target=colnames[-1]
predictors=colnames[:-1]

from sklearn.model_selection import train_test_split
train,test = train_test_split(dummy_data,test_size=0.2)

from sklearn.tree import DecisionTreeClassifier as DTC
model=DTC(criterion="entropy")
model.fit(train[predictors],train[target])

pred=model.predict(train[predictors])
train_acc=np.mean(pred==train[target]) #1.0

pred=model.predict(test[predictors])
test_acc=np.mean(pred==test[target]) # 0.6 overfiting
pd.crosstab(pred,test[target])

from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
 
bagging=BaggingClassifier(model,n_estimators=100)
bagging.fit(train[predictors],train[target])

bagging.score(train[predictors],train[target])  # 1.0
bagging.score(test[predictors],test[target])  #0.68

boosting=AdaBoostClassifier(model,n_estimators=50,learning_rate=0.01)
boosting.fit(train[predictors],train[target])

boosting.score(train[predictors],train[target])  # 1.0
boosting.score(test[predictors],test[target])  # 0.63


####################################################################

fraud=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/Fraud_check.csv")
fraud.head()
fraud.describe()

marital_status=[]
for i in fraud["Marital.Status"]:
    if i=='Single':
        marital_status.append(0)
    elif i=='Married':
        marital_status.append(1)
    else:
        marital_status.append(2)
        
fraud["Marital.Status"]=marital_status
fraud.head()

categorical_var=["Undergrad","Urban"]
fraud=pd.get_dummies(fraud,columns=categorical_var,drop_first=True)
fraud.head()

income_range=pd.cut(fraud["Taxable.Income"],bins=[0,30000,100000],labels=["Risky","Good"])
fraud["Taxable.Income"]=income_range
fraud.head()

colnames=list(fraud.columns)
target=colnames.pop(1)
predictors=colnames

train,test = train_test_split(fraud,test_size=0.2)
model=DTC(criterion="entropy")
model.fit(train[predictors],train[target])

pred=model.predict(train[predictors])
train_acc=np.mean(pred==train[target])  # 1.0

pred=model.predict(test[predictors])
test_acc=np.mean(pred==test[target])  # 0.65 overfitting

bagging=BaggingClassifier(model,n_estimators=15)
bagging.fit(train[predictors],train[target])

bagging.score(train[predictors],train[target])  # 0.98
bagging.score(test[predictors],test[target])  # 0.75 slightly increases

boosting=AdaBoostClassifier(model,n_estimators=70,learning_rate=0.001,)
boosting.fit(train[predictors],train[target])

boosting.score(train[predictors],train[target])  # 1.0
boosting.score(test[predictors],test[target])   #0.64
