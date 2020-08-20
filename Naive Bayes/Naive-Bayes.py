import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB,MultinomialNB

salary_train = pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\SalaryData_Train.csv")
salary_test = pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\SalaryData_Test.csv")
salary_train.describe()

salary_train.drop(columns=['educationno'],inplace=True)
salary_test.drop(columns=['educationno'],inplace=True)

plt.rcParams.update({'figure.figsize':(5,5),'figure.dpi':120})
salary_train['Salary'].value_counts() # dataset is imbalanced
sns.countplot(salary_train['Salary']) # more peaple with <50k salary

sns.boxplot(x='Salary',y='age',data=salary_train) # people earning >50k is more in age considering the middle 50% of the distribution 
sns.distplot(salary_train.age,bins=10) # more people having less age 
salary_train.age.skew() # positive skewed
salary_train.age.kurt() # non peaked distribution

sns.countplot(x='Salary',hue='sex',data=salary_train) # more men earn than women ie no of men>women in dataset
sns.countplot(x='Salary',hue='maritalstatus',data=salary_train) # most of poeple who earn >50k are married-civ-spouse, never married , widowed and separated generally earn <50k
sns.countplot(x='Salary',hue='workclass',data=salary_train) # more private employed
sns.countplot(x='Salary',hue='relationship',data=salary_train) # not in family,own child and unmarried generally earn <50k
sns.countplot(x='Salary',hue='race',data=salary_train) #white people are employed more

sns.catplot(x='Salary',y='age',hue='sex',kind='bar',data=salary_train) #age of men and women almost same for both categories
sns.catplot(x='Salary',y='age',hue='sex',kind='box',data=salary_train)

sns.catplot(x='Salary',y='hoursperweek',hue='sex',kind='bar',data=salary_train)
sns.catplot(x='Salary',y='hoursperweek',hue='sex',kind='box',data=salary_train) # men work slightly more hours than women for both categories

sns.catplot(x='Salary',y='hoursperweek',hue='maritalstatus',kind='bar',data=salary_train)
sns.catplot(x='Salary',y='age',hue='maritalstatus',kind='bar',data=salary_train) 
sns.catplot(x='Salary',y='hoursperweek',hue='race',kind='bar',data=salary_train)


colnames=salary_train.columns
cat_var=['workclass', 'education', 'maritalstatus', 'occupation','relationship', 'race', 'sex','native']
for i in cat_var:
    salary_train[i]=LabelEncoder().fit_transform(salary_train[i])
    salary_test[i]=LabelEncoder().fit_transform(salary_test[i])
     
predictors=colnames[:-1]
target=colnames[-1]

gau_nb=GaussianNB().fit(salary_train[predictors],salary_train[target])
gau_pred=gau_nb.predict(salary_train[predictors])
pd.crosstab(salary_train[target],gau_pred)
train_acc=np.mean(salary_train[target]==gau_pred) #79%

gau_pred=gau_nb.predict(salary_test[predictors])
pd.crosstab(salary_test[target],gau_pred)
test_acc=np.mean(salary_test[target]==gau_pred) #79%


mul_nb=MultinomialNB().fit(salary_train[predictors],salary_train[target])
mul_pred=mul_nb.predict(salary_train[predictors])
pd.crosstab(salary_train[target],mul_pred)
train_acc=np.mean(salary_train[target]==mul_pred) #77%

mul_pred=mul_nb.predict(salary_test[predictors])
pd.crosstab(salary_test[target],mul_pred)
test_acc=np.mean(salary_test[target]==mul_pred) #77%

# Naive Bayes generally not used where there are high number of predictors

###########################################################################

sms=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\sms_raw_NB.csv",encoding = "ISO-8859-1")
sms.type.value_counts() #imbalanced dataset
sms.columns

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

#clensing
for i in range(5559):
    words=re.sub('[^a-zA-Z]',' ',sms.text[i])
    words=words.lower()
    words=words.split()
    words=[lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    words=" ".join(words)
    sms.text[i]=words

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer().fit_transform(sms.text).toarray()
sms=pd.concat([sms,pd.DataFrame(tfidf)],axis=1)

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(sms.iloc[:,2:],sms.type,test_size=0.3)

gau_pred=GaussianNB().fit(train_x,train_y).predict(test_x)
pd.crosstab(gau_pred,test_y)
gau_acc=np.mean(gau_pred==test_y) #86.75%

mul_pred=MultinomialNB().fit(train_x,train_y).predict(test_x)
pd.crosstab(mul_pred,test_y)
mul_acc=np.mean(mul_pred==test_y) #96.22%








