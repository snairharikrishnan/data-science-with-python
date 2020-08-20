univ=read.csv(file.choose())
View(univ)
data<-univ[,-1]
cor(data)

pc<-princomp(data,cor = T,covmat = NULL,scores = T)
summary(pc)
str(pc)

plot(pc)
biplot(pc)

data<-cbind(data,pc$scores[,1:3])
View(data)

clus_data<-data[,7:9]
clus_data<-scale(clus_data)

distance<-dist(clus_data,method="euclidean")
fit<-hclust(distance,method="complete")

plot(fit,hang=-1)
rect.hclust(fit,5,border = "red")

group<-cutree(fit,5)
univ["Group"]<-group

aggregate(univ[,-c(1,8)],by=list(univ$Group),mean)
