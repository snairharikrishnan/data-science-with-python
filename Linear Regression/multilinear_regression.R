cars<-read.csv(file.choose())
View(cars)
attach(cars)

pairs(cars)
cor(cars)

install.packages("corpcor")
library(corpcor)
cor2pcor(cor(cars))

reg<-lm(MPG~., data = cars)
summary(reg)

regV<-lm(MPG~VOL, data = cars)
summary(regV)

regW<-lm(MPG~WT, data = cars)
summary(regW)

regVW<-lm(MPG~VOL+WT, data = cars)
summary(regVW)

plot(reg)  #various plots

influence.measures(reg)
influenceIndexPlot(reg)
influencePlot(reg)


reg1<-lm(MPG~., data = cars[-77,])
summary(reg1)

vif(reg)
avPlots(reg)

install.packages("MASS")
library(MASS)

stepAIC(reg)

reg_final<-lm(MPG~.-WT,data=cars[-77,])
summary(reg_final)

avPlots(reg_final)

vif(reg_final)
