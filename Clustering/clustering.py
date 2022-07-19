import pandas as pd
import numpy as np
import seaborn
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

crime=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\crime_data.csv")

def std_fun(data):
    x=(data-data.mean()) / data.std()
    return x

std_crime=std_fun(crime.iloc[:,1:])

h_clust=sch.linkage(std_crime,method="complete",metric="euclidean")

plt.figure(figsize=(15,7))
sch.dendrogram(h_clust,leaf_font_size=15.,)

h_clust=AgglomerativeClustering(n_clusters=4,linkage='complete',affinity='euclidean')
h_clust.fit(std_crime)

crime['label']=h_clust.labels_

crime.iloc[:,1:5].groupby(crime['label']).mean()

###########################################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

airline=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\EastwestAirlines.csv")

def std_fun(data):
    x=(data-data.mean()) / data.std()
    return x

std_air=std_fun(airline.iloc[:,1:])


#Scee plot
TWSS=[]
for i in range(2,15):
    km=KMeans(n_clusters=i).fit(std_air)
    WSS=[]
    for j in range(i):
        WSS.append(sum(cdist(std_air.iloc[km.labels_==j,:],km.cluster_centers_[j].reshape(1,std_air.shape[1]),"euclidean")))
    TWSS.append(sum(WSS))
    
plt.rcParams.update({'figure.figsize':(10,7)})
plt.plot(np.arange(2,15),TWSS,'bo-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS");plt.xticks(np.arange(2,15))

km=KMeans(n_clusters=6).fit(std_air)
km.labels_

airline['label']=km.labels_

mean_features=airline.iloc[:,1:-1].groupby(airline['label']).mean()








