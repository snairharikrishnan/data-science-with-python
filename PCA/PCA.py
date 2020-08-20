import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

wine = pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\wine.csv")

#scaling required
from sklearn.preprocessing import scale
std_wine=scale(wine.iloc[:,1:])

pca=PCA(n_components=13)
pca_values=pca.fit_transform(std_wine)

#Variance of each PC
var=pca.explained_variance_ratio_

cum_var=np.cumsum(var*100)
cum_var
plt.plot(cum_var,color="red")

TWSS=[]
for i in range(2,15):
    km=KMeans(n_clusters=i).fit(pca_values)
    WSS=[]
    for j in range(i):
        WSS.append(sum(cdist(pca_values[km.labels_==j],km.cluster_centers_[j].reshape(1,pca_values.shape[1]),'euclidean')))
    TWSS.append(sum(WSS))
    
plt.plot(np.arange(2,15),TWSS,'ro-')    
    
km=KMeans(n_clusters=3).fit(pca_values)
wine['label']=km.labels_    

mean_features=wine.iloc[:,1:-1].groupby(wine['label']).mean()
