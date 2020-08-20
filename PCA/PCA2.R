wine<-read.csv(file.choose())
View(wine)
summary(wine)

cor(wine[,-1])

pc<-princomp(wine[,-1],cor = T,scores = T,covmat = NULL)
summary(pc)  #  first three components captures 66.5% of the data
str(pc)

plot(pc)
biplot(pc)

wine<-cbind(wine,pc$scores[,1:3])
clus_data<-pc$scores[,1:3]
clus_data<-scale(clus_data)
View(clus_data)

distance<-dist(clus_data,method = "euclidean")
fit<-hclust(distance,method="complete")
plot(fit,hang=-1)
rect.hclust(fit,3,border = "red")  # three clusters will be inappropriate
rect.hclust(fit,5,border = "blue")

group<-cutree(fit,5)
wine<-cbind(group,wine)
#table(wine$Type,wine$group)

aggregate(wine[,3:15],by=list(wine$group),mean)


##################################################################

wssplot<-function(data,kmax){
  wss= (nrow(data)-1)*sum(apply(data,2,var))
  for(i in 2:kmax) wss[i]=sum(kmeans(data,centers=i)$withinss)
  plot(1:kmax,wss,type="b")
}

wssplot(clus_data,20)  # max slope at 2,3,6,7 clusters

library("kselection")
library("doParallel")

registerDoParallel(cores=4)
k <- kselection(clus_data, parallel = TRUE, k_threshold = 0.9, max_centers=12)
k  # f(k) finds 3 clusters

km<-kmeans(clus_data,3)
group<-km$cluster
wine<-cbind(group,wine)

confusion_matrix<-table(wine$Type,wine$group)
confusion_matrix
accuracy<-(confusion_matrix[1,2]+confusion_matrix[2,3]+confusion_matrix[3,1])/sum(confusion_matrix)
accuracy    #  97.19% accuracy
