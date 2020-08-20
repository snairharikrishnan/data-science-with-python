import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB,MultinomialNB

iris=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\iris.csv")
sns.scatterplot(x='Sepal.Length',y='Sepal.Width',hue='Species',data=iris)

from sklearn.model_selection import train_test_split
train,test=train_test_split(iris,test_size=0.2)

colnames=iris.columns
predictors=colnames[:-1]
target=colnames[-1]

gaussian_nb=GaussianNB().fit(train[predictors],train[target]).predict(test[predictors])
pd.crosstab(gaussian_nb,test[target])
gau_acc=np.mean(gaussian_nb==test[target])


multinomial_nb=MultinomialNB().fit(train[predictors],train[target]).predict(test[predictors])
pd.crosstab(multinomial_nb,test[target])
mul_acc=np.mean(multinomial_nb==test[target])






