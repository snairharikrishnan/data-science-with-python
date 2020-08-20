import pandas as pd
import numpy as np

iris = pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/iris.csv")
iris.head()

from sklearn.model_selection import train_test_split
train,test = train_test_split(iris,test_size=0.2)

train.head()

from sklearn.neighbors import KNeighborsClassifier as KNC

neigh=KNC(n_neighbors=3)
neigh.fit(train.iloc[:,0:4],train.iloc[:,4])

train_acc = np.mean(neigh.predict(train.iloc[:,0:4])==train.iloc[:,4])
test_acc = np.mean(neigh.predict(test.iloc[:,0:4])==test.iloc[:,4])

acc=[]

for i in range(3,50,2):
    neigh=KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:4],train.iloc[:,4]) 
    train_acc = np.mean(neigh.predict(train.iloc[:,0:4])==train.iloc[:,4])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:4])==test.iloc[:,4])
    acc.append((train_acc,test_acc))
    
from matplotlib import pyplot as plt
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-")
plt.legend(["train","test"])
