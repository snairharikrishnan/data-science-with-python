import pandas as pd
import numpy as np
import seaborn as sns

letters=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\letters.csv")
letters.head()

sns.boxplot(x="letter",y="x-box",data=letters,palette = "hls")
sns.boxplot(x="y-box",y="letter",data=letters,palette = "hls")

colnames=letters.columns
predictors=colnames[1:]
target=colnames[0]

from sklearn.model_selection import train_test_split
train,test=train_test_split(letters,test_size=0.2)

from sklearn.svm import SVC

# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

model=SVC(kernel="linear")
model=model.fit(train[predictors],train[target])
pred=model.predict(test[predictors])
acc=np.mean(pred==test[target])   # 84.8%

model=SVC(kernel="poly")
model=model.fit(train[predictors],train[target])
pred=model.predict(test[predictors])
acc=np.mean(pred==test[target])   # 94.72%

model=SVC(kernel="rbf")
model=model.fit(train[predictors],train[target])
pred=model.predict(test[predictors])
acc=np.mean(pred==test[target])   # 96.7%


