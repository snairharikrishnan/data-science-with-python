import pandas as pd
import numpy as np
import seaborn as sns

startup=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\50_Startups.csv")
startup.head()

# EDA
startup.describe()               # high Std dev and and min is zero for R&D and Marketing Spend
startup.skew()                   # negative skew for Administration and Marketing Spend 
startup.kurt()                   # positive kurtosis for Administration
startup['State'].value_counts()  # categorical variable not biased , balanced

startup.columns
startup.isnull().sum()  # No NA values
sns.lineplot(data=startup.drop(['State'],axis=1))

sns.distplot(startup['R&D Spend'],bins=10)       # almost normally distributed
sns.distplot(startup['Administration'],bins=10)  # negative skewed 
sns.distplot(startup['Marketing Spend'],bins=10) # almost normally distributed
sns.distplot(startup['Profit'],bins=10)          # almost normally distributed
startup.hist()

sns.boxplot(startup['R&D Spend'],color='yellow')        # no outliers
sns.boxplot(startup['Administration'],color='yellow')   # no outliers
sns.boxplot(startup['Marketing Spend'],color='yellow')  # no outliers
sns.boxplot(startup['Profit'],color='yellow')           # one outlier to the lower extreme
startup.boxplot()

sns.countplot(startup['State'])  # balanced

sns.scatterplot(x='R&D Spend',y='Profit',hue='State',data=startup)       # R&D and profit has a positive linear relationship
sns.scatterplot(x='Administration',y='Profit',hue='State',data=startup)  # random relationship
sns.scatterplot(x='Marketing Spend',y='Profit',hue='State',data=startup) # nearly positive linear relationship



startup.loc[startup.State=="California","State"]=0
startup.loc[startup.State=="Florida","State"]=1
startup.loc[startup.State=="New York","State"]=2
startup.head()

colnames=list(startup.columns)
predictors=colnames[:-1]
target=colnames[-1]

from sklearn.model_selection import train_test_split
train,test=train_test_split(startup,test_size=0.2)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler().fit(train)
train=scaler.transform(train)
test=scaler.transform(test)

train=pd.DataFrame(train,columns=colnames)
test=pd.DataFrame(test,columns=colnames)

from sklearn.neural_network import MLPRegressor
model=MLPRegressor(hidden_layer_sizes=(30,30),activation='relu')
model.fit(train[predictors],train[target])

pred=model.predict(train[predictors])
from sklearn.metrics import explained_variance_score
explained_variance_score(train[target],pred) # 97.85%
np.sqrt(np.mean((train[target]-pred)**2)) #RMSE 0.146

pred=model.predict(test[predictors])
explained_variance_score(test[target],pred) # 93.2%
np.sqrt(np.mean((test[target]-pred)**2))  #RMSE 0.215

#############################################################################

concrete=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\concrete.csv")
concrete.head()

# EDA

concrete.iloc[:,:5].describe()
concrete.iloc[:,5:].describe()
concrete.skew()  # coarseagg and fineagg are negatively skewed
concrete.kurt()  # water, superplastic, age have positive kurtosis, very high for age

concrete.isnull().sum() # no NA values
concrete.columns

sns.distplot(concrete['cement'],bins=10)         # almost normally distributed
sns.distplot(concrete['slag'],bins=10)           # positive skewed 
sns.distplot(concrete['ash'],bins=10)            # positive skewed
sns.distplot(concrete['water'],bins=10)          # negative skewed
sns.distplot(concrete['superplastic'],bins=10)   # positive skewed
sns.distplot(concrete['coarseagg'],bins=10)      # almost normally distributed
sns.distplot(concrete['fineagg'],bins=10)        # almost normally distributed
sns.distplot(concrete['age'],bins=10)            # positive skewed
sns.distplot(concrete['strength'],bins=10)       # almost normally distributed
concrete.hist()

sns.boxplot(concrete['cement'],color='green')         # no outliers
sns.boxplot(concrete['slag'],color='green')           # one outlier to the upper extreme
sns.boxplot(concrete['ash'],color='green')            # no outliers
sns.boxplot(concrete['water'],color='green')          # outliers on both extremes
sns.boxplot(concrete['superplastic'],color='green')   # two outliers on upper extreme
sns.boxplot(concrete['coarseagg'],color='green')      # no outliers
sns.boxplot(concrete['fineagg'],color='green')        # one outlier to the upper extreme
sns.boxplot(concrete['age'],color='green')            # outliers to the upper extreme
sns.boxplot(concrete['strength'],color='green')       # outliers to the upper extreme
concrete.boxplot()

sns.scatterplot(x='cement',y='strength',data=concrete)       # slight positive linear relationship
sns.scatterplot(x='slag',y='strength',data=concrete)         # random
sns.scatterplot(x='ash',y='strength',data=concrete)          # random
sns.scatterplot(x='water',y='strength',data=concrete)        # slight negative linear relationship
sns.scatterplot(x='superplastic',y='strength',data=concrete) # slight positive linear relationship
sns.scatterplot(x='coarseagg',y='strength',data=concrete)    # random
sns.scatterplot(x='fineagg',y='strength',data=concrete)      # random
sns.scatterplot(x='age',y='strength',data=concrete)          # random



colnames=list(concrete.columns)
predictors=colnames[:-1]
target=colnames[-1]

train,test=train_test_split(concrete,test_size=0.2)

scaler=StandardScaler().fit(train)
train=scaler.transform(train)
test=scaler.transform(test)

train=pd.DataFrame(train,columns=colnames)
test=pd.DataFrame(test,columns=colnames)

model=MLPRegressor(hidden_layer_sizes=(40,40,40),activation='relu')
model.fit(train[predictors],train[target])

pred=model.predict(train[predictors])
explained_variance_score(train[target],pred) #96.25%
np.sqrt(np.mean((train[target]-pred)**2)) #RMSE 0.199

pred=model.predict(test[predictors])
explained_variance_score(test[target],pred) # 90.8%
np.sqrt(np.mean((test[target]-pred)**2)) #RMSE 0.300

###############################################################################

fire=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\forestfires.csv")
fire.head()
fire["size_category"].unique()

fire.loc[fire.size_category=="small","size_category"]=0
fire.loc[fire.size_category=="large","size_category"]=1
fire.head()
fire["size_category"].unique()

fire.drop(["month","day"],axis=1,inplace=True)

x=fire.iloc[:,:-1]
y=fire.iloc[:,-1]

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2)

scaler=StandardScaler().fit(train_x)
train_x=scaler.transform(train_x)
test_x=scaler.transform(test_x)

from sklearn.neural_network import MLPClassifier
model=MLPClassifier(hidden_layer_sizes=(50,50,50),activation='relu')
model.fit(train_x,train_y)

pred=model.predict(train_x)
pd.crosstab(train_y,pred)
np.mean(train_y==pred)   #100%

pred=model.predict(test_x)
pd.crosstab(test_y,pred)
np.mean(test_y==pred)  #85.57%
