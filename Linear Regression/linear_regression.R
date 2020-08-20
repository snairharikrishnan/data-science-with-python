linear<-read.csv(file.choose())
View(linear)
install.packages("lattice")
library("lattice")
attach(linear)

dotplot(Waist)
dotplot(AT)

summary(linear)

plot(AT,Waist)

cor(Waist,AT)
reg<-lm(AT~Waist,data = linear)
summary(reg)

sqrt(mean(reg$residuals^2))

confint(reg,level=0.95)
pred<-predict(reg,interval = "predict")
View(pred)

linear<-cbind(linear,pred)

install.packages("ggplot2")
library("ggplot2")

?ggplot

ggplot(data = linear,aes(x=Waist,y=AT))+ geom_point(color="blue")+geom_line(color="red",data = linear,aes(x=Waist, y=pred))

############   log model   ################

plot(log(Waist),AT)
cor(log(Waist),AT)

reg_log<-lm(AT~log(Waist),data = linear)
summary(reg_log)

sqrt(mean(reg_log$residuals^2))

confint(reg_log,level=0.95)
pred_log<-predict(reg_log,interval="confidence")
View(pred_log)

linear<-cbind(linear[,1:2],pred_log)
View(linear)


ggplot(data = linear,aes(x=Waist,y=AT))+geom_point(color="red")+geom_line(color="blue",data = linear,aes(x=Waist,y=pred_log))

#########  exponential  #############

plot(Waist,log(AT))
cor(Waist,log(AT))

reg_exp<-lm(log(AT)~Waist,data = linear)
summary(reg_exp)

sqrt(mean(reg_exp$residuals^2))

confint(reg_exp,level=0.95)
pred_exp<-predict(reg_exp)
View(pred_exp)

linear<-cbind(linear[,1:2],exp(pred_exp))
View(linear)


ggplot(data = linear,aes(x=Waist,y=AT))+geom_point(color="red")+geom_line(color="blue",data = linear,aes(x=Waist,y=exp(pred_exp)))

#########   polynomial   ##########

plot(Waist+Waist^2,AT)
cor(Waist+Waist^2,AT)

reg_pol<-lm(AT~Waist+I(Waist*Waist),data = linear)
summary(reg_pol)

sqrt(mean(reg_pol$residuals^2))

confint(reg_pol,level=0.95)
pred_pol<-predict(reg_pol)
View(pred_exp)

linear<-cbind(linear[,1:2],exp(pred_exp))
View(linear)


ggplot(data = linear,aes(x=Waist+I(Waist*Waist),y=AT))+geom_point(color="red")+geom_line(color="blue",data = linear,aes(x=Waist+I(Waist*Waist),y=pred_pol))
