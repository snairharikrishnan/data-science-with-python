import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier as KNC

zoo=pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/Zoo.csv")
zoo.head()
zoo.shape
zoo.columns
zoo.isna().sum() # no NA values

#EDA
plt.rcParams.update({'figure.figsize':(7,5),'figure.dpi':120})
sns.countplot(x='type',data=zoo) # type 1 animals more
sns.countplot(x='type',hue='hair',data=zoo) # hair only for type 1 and 6
sns.countplot(x='type',hue='feathers',data=zoo) # feathers only for type 2
sns.countplot(x='type',hue='eggs',data=zoo)# no eggs most of type1 and small fraction type 3 and 7
sns.countplot(x='type',hue='milk',data=zoo) # milk only for type 1
sns.countplot(x='type',hue='airborne',data=zoo)# few in type1, majority in type 2 and 6 are airborne
sns.countplot(x='type',hue='aquatic',data=zoo)#all of type 4 and 5 purely aquatic;6 purely non aquatic
sns.countplot(x='type',hue='predator',data=zoo)#type 2 and 6 mostly non predators while rest mostly predators
sns.countplot(x='type',hue='toothed',data=zoo)#all of type 2,6,7 are non toothed,all of type 4,5 are toothed while most of type 1,3 are toothed
sns.countplot(x='type',hue='backbone',data=zoo)# no backbone for type 6 and 7 while rest has backbone
sns.countplot(x='type',hue='breathes',data=zoo)#all under type 1,2,5,6 breathes, while all under type 4 doesnt breathe and maajority of type 3 and minority of type7 breathes 
sns.countplot(x='type',hue='venomous',data=zoo)# all under 1,2 non venemous,mojority within rest groups non venemous 
sns.countplot(x='type',hue='fins',data=zoo)#all under type 2,3,5,6,7 doesnt have fins,all under type 4 has fins, majority of type1 doesnt have fins
sns.countplot(x='type',hue='legs',data=zoo)#type1 has 0,2,4 legged species(mojority 4 legged),type2 all 2 legged,type3 0 and 4 legged,type4 all 0 legged,type5 all 4legged, type6 all 6 legged, type7 mix of 0,4,5,6,8 legged
sns.countplot(x='type',hue='tail',data=zoo)# type 2,3,4 has tail, type6 no tail,majority of type1 tailed, mojority of type5,7 non tailed 
sns.countplot(x='type',hue='domestic',data=zoo)#type 3,5,7 non tailed,majority of type 1,2,4,6 non tailed
sns.countplot(x='type',hue='catsize',data=zoo)#majority of type 2,3,4,7 catsize0, majority of type1 catsize1, type 5 and 6 catsize0


neigh=KNC(n_neighbors=3)
neigh.fit(zoo.iloc[:,1:17],zoo.iloc[:,17])
accuracy=np.mean(neigh.predict(zoo.iloc[:,1:17])==zoo.iloc[:,17])

acc=[]
for i in range(1,10,1):
    neigh=KNC(n_neighbors=i)
    neigh.fit(zoo.iloc[:,1:17],zoo.iloc[:,17])
    accuracy=np.mean(neigh.predict(zoo.iloc[:,1:17])==zoo.iloc[:,17])
    acc.append(accuracy)
    
plt.plot(np.arange(1,10,1),[i for i in acc],"bo-")

# We get 100% accuracy when we try with 1 or 2 nearest neighbour

neigh=KNC(n_neighbors=2)
neigh.fit(zoo.iloc[:,1:17],zoo.iloc[:,17])
zoo["predicted"]=neigh.predict(zoo.iloc[:,1:17])

#splitting data on training and test
from sklearn.model_selection import train_test_split
train,test=train_test_split(zoo,test_size=0.2)

acc=[]

for i in range(1,10,1):
    neigh=KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,1:17],train.iloc[:,17])
    train_acc=np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])
    test_acc=np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])
    acc.append((train_acc,test_acc))
    

plt.plot(np.arange(1,10,1),[i[0] for i in acc],"bo-")
plt.plot(np.arange(1,10,1),[i[1] for i in acc],"ro-")
plt.legend("train","test")

#When we use 1-NN we get 100% accuracy on training as well as test data

##########################################################################

glass = pd.read_csv("C:/Users/snair/Documents/Data Science Assignment/Data Sets/glass.csv")
glass.head()
glass.describe()

acc=[]
for i in range(1,10,1):
    neigh=KNC(n_neighbors=i)
    neigh.fit(glass.iloc[:,:-1],glass.iloc[:,-1])
    accuracy=np.mean(neigh.predict(glass.iloc[:,:-1])==glass.iloc[:,-1])
    acc.append(accuracy)

plt.plot(np.arange(1,10,1),[i for i in acc],"bo-")

# We get 100% accuracy when we try with 1 nearest neighbour
neigh=KNC(n_neighbors=1)
neigh.fit(glass.iloc[:,:-1],glass.iloc[:,-1])
glass["predicted"]=neigh.predict(glass.iloc[:,:-1])
glass.head()







