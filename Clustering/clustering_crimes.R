##################  Heirarchical Clustering  ###################

crimes<-read.csv(file.choose())
View(crimes)
attach(crimes)

summary(crimes)
boxplot(Rape,horizontal=T)
boxplot(Assault,horizontal=T)
boxplot(Murder,horizontal=T)
boxplot(UrbanPop,horizontal=T)


std_crimes<-scale(crimes[,-c(1)])
View(std_crimes)

distance<-dist(std_crimes,method = "euclidean")
fit<-hclust(distance,method = "complete")

plot(fit,hang = -1)

rect.hclust(fit,k=4,border = "blue")
group<-cutree(fit,k=4)

crimes["Group"]<-group

?aggregate()
aggregate(crimes[,-1],by=list(crimes$Group),mean)

#  Inference -->  group 2 has highest crime rate then 1 then 3 and last 4

##################  K-means ############################
crimes<-read.csv(file.choose())
View(crimes)
attach(crimes)

std_crimes<-scale(crimes[,-c(1)])

wssplot<-function(data,kmax=10){
  wss<-(nrow(data)-1)*sum(apply(data,2,var))
  for(i in 2:kmax) wss[i]=sum(kmeans(data,centers = i)$withinss)
  plot(1:kmax,wss,type = "b")
}

twss <- NULL
twssplot<-function(standardized_data,kmax=15){
  for (i in 2:kmax){
    twss <- c(twss,kmeans(standardized_data,i)$tot.withinss)
  }
  plot(2:kmax,twss,type="o")
}

wssplot(std_crimes,kmax=10)
twssplot(std_crimes,kmax=15)

km<-kmeans(std_crimes,3)
str(km)
crimes["groups"]<-km$cluster

aggregate(crimes[,-1],by=list(crimes$groups),mean)

#Inference --> group 2 has highest crime rate then group 1, then 3