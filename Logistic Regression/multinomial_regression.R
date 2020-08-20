install.packages("mlogit")
install.packages("nnet")
library(nnet)
library(mlogit)
data("Mode")
View(Mode)
head(Mode)
attach(Mode)

table(Mode$choice)
reg<-multinom(choice~.,data = Mode)
summary(reg)

Mode$choice<-relevel(Mode$choice,ref = "carpool")  #to change the baseline level

####   To get p-value   ####
z<-summary(reg)$coefficients/summary(reg)$standard.errors
p_value<-(1-pnorm(abs(z),0,1))*2
p_value

####   Odds Ratio   ####
exp(coef(reg))  #coef is logit function

prob<-fitted(reg)
prob<-as.data.frame(prob)
View(prob)

get_name<-function(i){
  return(names(which.max(i)))
}

?apply
pred_name<-apply(prob,1,get_name)
prob["pred"] <- pred_name

table(pred_name,Mode$choice)
?barplot
barplot(table(pred_name,Mode$choice),beside = T,col=c("red","green","blue","orange"),legend=c("bus","car","carpool","rail"),main = "Predicted(X-axis) - Legends(Actual)",ylab = "count")

mean(pred_name==Mode$choice)   #  69.31%
