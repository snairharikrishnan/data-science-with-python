Contract<-read.csv(file.choose())
View(Contract)
attach(Contract)

qqnorm(Supplier.A)
qqnorm(Supplier.B)
qqnorm(Supplier.C)

shapiro.test(Supplier.A)
shapiro.test(Supplier.B)
shapiro.test(Supplier.C)

#p high null fly
#all are normal
 
var.test(Supplier.A,Supplier.B,Supplier.C) #error

boxplot(Supplier.A,Supplier.B,Supplier.C)

Stacked_Data<-stack(Contract)
View(Stacked_Data)

#unstack(Stacked_Data)

attach(Stacked_Data)

?aov

Anova_result<-aov(values~ind,data=Stacked_Data)
summary(Anova_result)
#p high null fly
#Renew all contracts