
library(car)
library(dplyr)

corolla<-read.csv(file.choose())
View(corolla)
attach(corolla)

corolla<-corolla[,c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]
summary(corolla)

pairs(corolla)
cor(corolla)

reg<-lm(Price~.,data = corolla)
summary(reg)   # CC and Doors insignificant

regc<-lm(Price~cc,data = corolla)
summary(regc)   # cc significant independently

regd<-lm(Price~Doors,data = corolla)
summary(regd)   # Doors significant independently

regcd<-lm(Price~cc+Doors,data = corolla)
summary(regcd)   # cc+Doors significant independently

summary(corolla)
boxplot(cc,horizontal = T)
boxplot(KM,horizontal = T)

plot(reg)  # KM min->1, cc max->16000
max(corolla[,"cc"])

corolla1<-filter(corolla,cc < max(corolla[,"cc"]))

filter(corolla,KM < 5)

corolla1<-filter(corolla,KM != 1)
boxplot(corolla1$KM,horizontal = T)

attach(corolla1)
View(corolla1)

reg<-lm(Price~.,data = corolla1)
summary(reg)

plot(reg)

influenceIndexPlot(reg)
vif(reg)
avPlots(reg)

corolla1<-corolla1[-c(81,954,216),]  # removing influencing records

reg<-lm(Price~.,data = corolla1)
summary(reg) #  all variables significant   R^2 = 0.8855
cor(corolla1)

sqrt(mean(reg$residuals^2))

pred<-predict(reg)

corolla2<-corolla1[,-1]
corolla2<-cbind(corolla2,corolla1[,1])
corolla2<-cbind(corolla2,pred)
View(corolla2)
names(corolla2)[9]<-"Price"
names(corolla2)[10]<-"Predicted_Price"

corolla2<-mutate(corolla2,Error=abs(Predicted_Price-Price))
corolla2<-mutate(corolla2,Error_Percent=(Error/Price)*100)


############################################################

startup<-read.csv(file.choose())
View(startup)
attach(startup)
summary(startup)

boxplot(R.D.Spend,horizontal = T)
boxplot(Administration,horizontal = T)
boxplot(Marketing.Spend,horizontal = T)
boxplot(Profit,horizontal = T)

pairs(startup)
cor(startup)

install.packages("dummies")
library(dummies)

startup<-dummy.data.frame(startup,sep=".",drop=T)
?dummy.data.frame
View(startup)
names(startup)[6]<-"State.New_York"
cor(startup)
pairs(startup)

reg<-lm(Profit~R.D.Spend+Administration+Marketing.Spend+State.California+State.Florida+State.New_York,data = startup)
summary(reg)
plot(reg)

avPlots(reg)
influenceIndexPlot(reg)

startup<-startup[-c(45,46),]

plot(Administration,Profit)
plot(R.D.Spend,Profit)
plot(Marketing.Spend,Profit)
plot(State.California,Profit)
plot(State.Florida,Profit)
plot(`State.New York`,Profit)

startup<-filter(startup,R.D.Spend&Marketing.Spend!=0)
View(startup)

reg1<-lm((Profit)~sqrt(R.D.Spend)+sqrt(Administration)+sqrt(Marketing.Spend^19),data = startup)
summary(reg1)  # R^2 = 0.9346

pred<-predict(reg1)
startup<-cbind(startup,pred)
View(startup)

startup<-mutate(startup,Error=abs(pred-Profit))
startup<-mutate(startup,Error_Percent=(Error/Profit)*100)

#################################################################################

computer<-read.csv(file.choose())
View(computer)
attach(computer)
summary(computer)

boxplot(price,horizontal = T)
boxplot(hd,horizontal = T)
boxplot(ads,horizontal = T)

computer<-mutate(computer,cd1=ifelse(cd=="yes",1,0))
computer<-mutate(computer,multi1=ifelse(multi=="yes",1,0))
computer<-mutate(computer,premium1=ifelse(premium=="yes",1,0))

#computer<-dummy.data.frame(computer,sep=".")

computer<-computer[,-c(1,7,8,9)]
View(computer)

pairs(computer)
cor(computer)
plot(speed,price)
plot(hd,price)

reg<-lm(price~.,data = computer)
summary(reg)   # R^2 = 0.7752

plot(reg)
influenceIndexPlot(reg)
avPlots(reg)

reg<-lm(log(price)~log(speed)+log(hd)+(ram)+(screen)+(ads)+(trend)+(cd)+multi+premium,data = computer)
summary(reg)  # R^2 = 0.8143

pred<-predict(reg)
pred<-exp(pred)

computer<-cbind(computer,pred)
View(computer)
computer<-mutate(computer,Error=abs(pred-price))
computer<-mutate(computer,Error_Percent=(Error/price)*100)
