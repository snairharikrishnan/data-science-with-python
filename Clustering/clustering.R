univ<-read.csv(file.choose())
View(univ)
attach(univ)
summary(univ)

standardized_data<-scale(univ[,-1])
View(standardized_data)

d<-dist(standardized_data,method = "euclidean")

fit<-hclust(d,method = "complete")
plot(fit)
plot(fit,hang = -1)

rect.hclust(fit,k=3,border="red")

groups<-cutree(fit,k=3)
univ["group"]<-groups

#################################

wssplot<-function(standardized_data,kmax=8){
wss = (nrow(standardized_data)-1)*sum(apply(standardized_data, 2, var))		 # Determine number of clusters by scree-plot 
for (i in 2:kmax) wss[i] = sum(kmeans(standardized_data, centers=i)$withinss)
 plot(1:kmax, wss, type="b", xlab="Number of Clusters", ylab="Within groups sum of squares")   # Look for an "elbow" in the scree plot #
title(sub = "K-Means Clustering Scree-Plot")}


# Creating a empty variable to store total within sum of sqares of clusters
twss <- NULL
twssplot<-function(standardized_data,kmax=15){
for (i in 2:kmax){
  twss <- c(twss,kmeans(standardized_data,i)$tot.withinss)
}
  plot(2:kmax,twss,type="o")
}

wssplot(standardized_data,kmax=8)
twssplot(standardized_data,kmax=15)
  
km<-kmeans(standardized_data,4) #kmeans clustering
str(km)
univ["groups"] <- NULL
univ$groups <- km$cluster



install.packages("kselection")
library(kselection)
?kselection

# To implement the parallel processing in getting the optimum number of
# clusters using kselection method 
install.packages("doParallel")
library(doParallel)
registerDoParallel(cores=4)
k <- kselection(univ[,-1], parallel = TRUE, k_threshold = 0.9, max_centers=12)
k
