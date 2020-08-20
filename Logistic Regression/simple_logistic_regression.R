claimants<-read.csv(file.choose())
View(claimants)
attach(claimants)
summary(claimants)

claimants<-na.omit(claimants)
claimants<-claimants[,-1]
library("dplyr")
plot(CLMAGE,ATTORNEY)
filter(claimants,CLMAGE==0)
claimants<-filter(claimants,CLMAGE!=0)
claimants<-filter(claimants,LOSS!=0)

reg<-glm(ATTORNEY~.,data = claimants,family = "binomial")
summary(reg)



reg<-glm(ATTORNEY~sqrt(CLMAGE)+log(LOSS)+CLMSEX+CLMINSUR,data = claimants,family = "binomial")
summary(reg)

library(car)
vif(reg)
influenceIndexPlot(reg)
avPlots(reg)

plot(reg)
claimants<-claimants[-293,]
claimants<-filter(claimants,LOSS<max(LOSS))

qqnorm(LOSS)
qqline(LOSS)
boxplot(LOSS,horizontal = T)


pred<-predict(reg,data=claimants,type = "response")
summary(pred)
View(pred)
contigency<-table(pred>0.5,claimants$ATTORNEY)
contigency

p_value<-sum(diag(contigency))/sum(contigency)
p_value

pred_value<-ifelse(pred>=0.5,1,0)
yes_no<-ifelse(pred>=0.5,"YES","NO")
claimants<-cbind(claimants,pred_value)
claimants<-cbind(claimants,yes_no)
View(claimants)
