setwd("C:/Users/HARIKRISHNAN/Downloads")
getwd()

gc()    #release ram

Promotion<-read.csv(file.choose())
attach(Promotion)   # attaches table to ram , so no need mention everytime

View(Promotion)
 
colnames(Promotion)<- c("Credit","Type","InterestRateWaiver","StandardPromotions")

View(Promotion)


shapiro.test(InterestRateWaiver)  #normality test
 # p = 0.2246     p>0.05  p high null fly  H0 holds ie normal

shapiro.test(StandardPromotions)
# p = 0.1916    normal


qqnorm(InterestRateWaiver)
qqnorm(StandardPromotions)

var.test(InterestRateWaiver,StandardPromotions) # variance test
# p = 0.653  p high null fly ie variance are equal


##### go for 2 sample T test for equal variances  #####

?t.test
# t.test used for both 1 sample and 2 sample
t.test(InterestRateWaiver,StandardPromotions,alternative = "two.sided",conf.level = 0.95,correct=TRUE)

#alternative=two sided means we are checking for both equal and unequal variance..already proven equal
# null hypothesis       <- Equal means
#alternate hypothesis   <- Unequal 
# p low null go
# so purachses made due to IntWaiver and StdPromotion not equal
# we are interested to prove IntWaiver > StdPromotions

t.test(InterestRateWaiver,StandardPromotions,alternative = "greater",conf.level = 0.95,correct=TRUE)
# null hypothesis       <- IntWaiver < StdPromotion
# alternate hypothesis  <- IntWaiver > StdPromotion 
# p low null go
# go for the Interest Rate Waiver


