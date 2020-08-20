import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from sklearn.model_selection import cross_val_score

bank=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\bank-full (1).csv")
bank.describe
bank.columns
bank.isna().sum() #no null values but many unknown values
plt.rcParams.update({'figure.figsize':(7,5),'figure.dpi':120})

sns.countplot(bank.y) #imbalanced dataset
bank.y.value_counts()

bank.job.value_counts()
sns.countplot(bank.job)

bank.marital.value_counts()
sns.countplot(bank.marital) # more married

bank.education.value_counts() # secondary educated more
sns.countplot(bank.education)

bank.housing.value_counts() # more people with housing loan
sns.countplot(bank.housing)

bank.loan.value_counts() # very less people with personal loan
sns.countplot(bank.loan)

bank.contact.value_counts() # secondary educated more
sns.countplot(bank.contact) #more people with cellular contact

pd.crosstab(bank.job,bank.y).plot()
pd.crosstab(bank.marital,bank.y).plot()
pd.crosstab(bank.default,bank.y).plot()
pd.crosstab(bank.housing,bank.y).plot() # housing loan reduces chance of subscription
pd.crosstab(bank.loan,bank.y).plot() 
pd.crosstab(bank.contact,bank.y).plot() #celluar contact can increase chance of subscription
pd.crosstab(bank.poutcome,bank.y).plot() #doesn't really matter


sns.catplot(x='y',y='age',kind='bar',data=bank)
sns.catplot(x='y',y='balance',kind='bar',data=bank) #more balance with people who has subscribed to term deposit
sns.catplot(x='y',y='duration',kind='bar',data=bank) #more duration of call more chance of subscription

sns.catplot(x='y',y='balance',hue='job',kind='bar',data=bank)
sns.catplot(x='y',y='balance',hue='marital',kind='bar',data=bank)
sns.catplot(x='y',y='balance',hue='housing',kind='bar',data=bank)
sns.catplot(x='y',y='balance',hue='loan',kind='bar',data=bank)

# Normality Check
plt.subplot(1,2,1)                                      #boxcox
sns.distplot(bank['age'],bins=15)
plt.subplot(1,2,2)
stats.probplot(bank['age'],dist="norm",plot=plt)

plt.subplot(1,2,1)
sns.distplot(bank['balance'],bins=20)
plt.subplot(1,2,2)
stats.probplot(bank['balance'],dist="norm",plot=plt)

plt.subplot(1,2,1)
sns.distplot(bank['day'],bins=20)
plt.subplot(1,2,2)
stats.probplot(bank['day'],dist="norm",plot=plt)

plt.subplot(1,2,1)                      #log
sns.distplot(bank['duration'],bins=20)
plt.subplot(1,2,2)
stats.probplot(bank['duration'],dist="norm",plot=plt)

plt.subplot(1,2,1)                      #SQRT
sns.distplot(bank['campaign'],bins=20)
plt.subplot(1,2,2)
stats.probplot(bank['campaign'],dist="norm",plot=plt)

# None of the continous variables are normally distributed

def  transform(feature,typ):  
    if typ=='log':
        feature=np.log(feature+1)
    elif typ=='recp':
        feature=1/(feature+1)
    elif typ=='sqrt':
        feature=feature**0.5
    elif typ=='exp':
        feature=feature**0.2
    else:
        feature,param=stats.boxcox(feature+1)

    plt.subplot(1,2,1)
    sns.distplot(feature,bins=15)
    plt.subplot(1,2,2)
    stats.probplot(feature,dist="norm",plot=plt)

transform(bank['balance'],'log')
transform(bank['age'],'log')            
transform(bank['duration'],'log')   



bank['age']=np.log(bank['age'])             # logarithmic transformation of age
bank['duration']=np.log(bank['duration']+1) # logarithmic transformation of duration
bank['campaign']=bank['campaign']**0.5      # Exponential transformation of campaign


cat_var=['job','marital','education','default','housing','loan','contact','poutcome']
for i in cat_var:                                 # finding number unknown values
    x=bank[i].value_counts().index
    if 'unknown' in x:
        na=bank[i].value_counts()['unknown']
        print(f"{(i,na,na*100/len(bank))}")
# 28% of contact and 81% of poutcome are unknown values


#Imputation
x1=0;x2=0
imp=['job','education','contact']
for i in range(45211):             # imputing unknown values with alternating top 2 values for job and education
    if bank[imp[0]][i]=='unknown':
        if x1%2==0:
            bank[imp[0]][i]='blue-collar'
            x1+=1
        else:
            bank[imp[0]][i]='management'
    if bank[imp[1]][i]=='unknown':
        if x2%2==0:
            bank[imp[1]][i]='secondary'
            x2+=1
        else:
            bank[imp[1]][i]='tertiary'
    if bank[imp[2]][i]=='unknown':
            bank[imp[2]][i]='cellular'   # imputating with mode value

#poutcome can be dropped since 81% unknown
bank.drop(['poutcome'],axis=1,inplace=True)

bank.pdays.unique()
for i in range(len(bank)):   # replacing -1 of pdays with 0
    if bank.pdays[i]==-1:
        bank.pdays=0
        
        
cat_var=['job','marital','education','default','housing','loan','contact','month','y']
cat=pd.get_dummies(bank[cat_var],drop_first=True) #creating dummy variables
bank.drop(bank[cat_var],axis=1,inplace=True)
bank=pd.concat([bank,cat],axis=1)

X=bank.iloc[:,:-1]
Y=bank.iloc[:,-1]


#OverSampling to convert imbalanced dataset to balanced
smk=SMOTETomek(random_state=0)
X,Y=smk.fit_sample(X,Y)        
X.shape          
Y.shape       #balanced the dataset

#Feature Selection
feature_importance_model=ExtraTreesClassifier()
feature_importance_model.fit(X,Y)
score=list(feature_importance_model.feature_importances_)

col=list(bank.columns)
col.pop()
score_dict={col[i]:score[i] for i in range(len(col))}
{k: v for k, v in sorted(score_dict.items(), key=lambda item: item[1],reverse=True)}

#removing pdays as score is 0
X.drop(['pdays'],axis=1,inplace=True)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=25)

model=LogisticRegression()
model.fit(x_train,y_train)
pred=model.predict(x_train)
pd.crosstab(pred,y_train)
np.mean(pred==y_train)   # 88% train accuracy

pred=model.predict(x_test)
pd.crosstab(pred,y_test)
np.mean(pred==y_test)  # 86% test accuracy

val_score=cross_val_score(model,X,Y,cv=100)
val_score.mean() # average accuracy of 88%

###########################################################################

credit=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\creditcard.csv")
credit.describe
credit=credit.iloc[:,1:]  #removing index column
credit.isna().sum()  # no NA values

sns.countplot(credit.card) # imbalanced
credit.card.value_counts()

sns.countplot(credit.owner)
credit.owner.value_counts()

sns.countplot(credit.selfemp)
credit.selfemp.value_counts()

sns.catplot(x='card',y='age',kind='bar',data=credit)
sns.catplot(x='card',y='reports',kind='bar',data=credit)
sns.catplot(x='card',y='income',kind='bar',data=credit)
sns.catplot(x='card',y='share',kind='bar',data=credit)
sns.catplot(x='card',y='expenditure',kind='bar',data=credit)
credit['expenditure'][credit['card']=='no'].sum()   # if credit card expenditure is 0, then card wont be issues
sns.catplot(x='card',y='dependents',kind='bar',data=credit)
sns.catplot(x='card',y='months',kind='bar',data=credit)
sns.catplot(x='card',y='majorcards',kind='bar',data=credit)
sns.catplot(x='card',y='active',kind='bar',data=credit)

cat_var=['owner','selfemp','card']
dummy=pd.get_dummies(credit[cat_var],drop_first=True) #creating dummy variables

credit.drop(cat_var,axis=1,inplace=True)
credit=pd.concat([credit,dummy],axis=1)

X=credit.iloc[:,:-1]
Y=credit.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=10)

model=LogisticRegression()
model.fit(x_train,y_train)
pred=model.predict(x_train)
pd.crosstab(pred,y_train)
np.mean(pred==y_train) #98% training accuracy

pred=model.predict(x_test)
pd.crosstab(pred,y_test)
np.mean(pred==y_test) #97.5% test accuracy

val_score=cross_val_score(model,X,Y,cv=10)
val_score.mean() # mean test accuracy of 98.1%
