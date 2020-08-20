
######  simple linear regression  #########

calories<-read.csv(file.choose())
View(calories)
names(calories)[1]<-"Weight.Gained"
attach(calories)

plot(Calories.Consumed,Weight.Gained)

cor(Calories.Consumed,Weight.Gained)

reg<-lm(Weight.Gained~Calories.Consumed,data = calories)  #  R^2 = 0.8882 
summary(reg)

confint(reg,level = 0.95)
pred<-predict(reg,interval = "confidence")
View(pred)

calories<-cbind(calories,pred[,1])
names(calories)[3]<-"linear_pred"
sqrt(mean(reg$residuals^2))

library(ggplot2)

ggplot(data = calories,aes(x=Calories.Consumed,y=Weight.Gained))+
  geom_point(color="red")+
  geom_line(color="blue",data = calories,aes(x=Calories.Consumed,y=pred[,1]))

##   square   ##

reg_sq<-lm(Weight.Gained~I(Calories.Consumed^2),data = calories)  #  R^2 = 0.9382 
summary(reg_sq)

confint(reg_sq,level = 0.95)
pred_sq<-predict(reg_sq,interval = "confidence")
View(pred_sq)

calories<-cbind(calories,pred_sq[,1])
names(calories)[4]<-"square_pred"
View(calories)

sqrt(mean(reg_sq$residuals^2))

ggplot(data = calories,aes(x=Calories.Consumed,y=Weight.Gained))+
  geom_point(color="red")+
  geom_line(color="blue",data = calories,aes(x=Calories.Consumed,y=pred_sq[,1]))

###   polinomial   ###

reg_pol<-lm(Weight.Gained~I(Calories.Consumed+Calories.Consumed^2+Calories.Consumed^3),data = calories)
summary(reg_pol)   #  R^2 = 0.9384 

confint(reg_pol,level = 0.95)
pred_pol<-predict(reg_pol,interval = "confidence")
View(pred_pol)

calories<-cbind(calories,pred_pol[,1])
names(calories)[5]<-"polinomial_pred"
View(calories)

sqrt(mean(reg_pol$residuals^2))

ggplot(data = calories,aes(x=Calories.Consumed,y=Weight.Gained))+
  geom_point(color="red")+
  geom_line(color="blue",data = calories,aes(x=Calories.Consumed,y=pred_pol[,1]))


######################################################################

delivery<-read.csv(file.choose())
View(delivery)
attach(delivery)

plot(Sorting.Time,Delivery.Time)

cor(Sorting.Time,Delivery.Time)

reg<-lm(Delivery.Time~Sorting.Time,data = delivery)  #  R^2 = 0.6655
summary(reg)

reg<-lm(Delivery.Time~log(Sorting.Time),data = delivery)#  R^2 = 0.6794
summary(reg)

reg<-lm(log(Delivery.Time)~Sorting.Time,data = delivery)  #  R^2 = 0.6957
summary(reg)

pred<-predict(reg)
pred<-exp(pred)

delivery<-cbind(delivery,pred)
View(delivery)

ggplot(data = delivery,aes(x=Sorting.Time,y=Delivery.Time))+
  geom_point(color="red")+
  geom_line(color="blue",data = delivery,aes(x=Sorting.Time,y=pred))

##########################################################

emp<-read.csv(file.choose())
View(emp)
attach(emp)
plot(Salary_hike,Churn_out_rate)

cor(Salary_hike,Churn_out_rate)

reg<-lm(Churn_out_rate~Salary_hike,data = emp) #  R^2 = 0.8101
summary(reg)

pred<-predict(reg)
View(pred)
emp<-cbind(emp,pred)
View(emp)


ggplot(data = emp,aes(x=Salary_hike,y=Churn_out_rate))+
  geom_point(color="red")+
  geom_line(color="blue",data = emp,aes(x=Salary_hike,y=pred))

min(Churn_out_rate)

reg_exp<-lm(-log(Churn_out_rate-58.3)~Salary_hike,data = emp) # R^2 = 0.9947
summary(reg_exp)
confint(reg_exp,level = 0.95)

pred_exp<-predict(reg_exp)
pred_exp<-58.3+exp(-pred_exp)
View(pred_exp)

emp<-cbind(emp,pred_exp)
View(emp)

emp<-emp[,-4]

ggplot(data = emp,aes(x=Salary_hike,y=Churn_out_rate))+
  geom_point(color="red")+
  geom_line(color="blue",data = emp,aes(x=Salary_hike,y=pred_exp))

#######################################################

sal<-read.csv(file.choose())
View(sal)
attach(sal)

plot(YearsExperience,Salary)

cor(YearsExperience,Salary)

reg<-lm(Salary~YearsExperience,data = sal)  # R^2 = 0.9554
summary(reg)

sqrt(mean(reg$residuals^2))

confint(reg,level = 0.95)

pred<-predict(reg)

sal<-cbind(sal,pred)
View(sal)

ggplot(data = sal,aes(x=YearsExperience,y=Salary))+
  geom_point(color="red")+
  geom_line(color="blue",data=sal,aes(x=YearsExperience,y=pred))

################################################################