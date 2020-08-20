
############   2 Sample T Test   ##############

cutlets<-read.csv(file.choose())
View(cutlets)
attach(cutlets)

boxplot(Unit.A)
boxplot(Unit.B)

#normality test
shapiro.test(Unit.A)
qqnorm(Unit.A)
qqline(Unit.A)

shapiro.test(Unit.B)
qqnorm(Unit.B)
qqline(Unit.B)

#variance test
var.test(Unit.A,Unit.B)

#2 sample t assuming equal variance
t.test(Unit.A,Unit.B,var.equal=TRUE)


############   One way ANOVA   #################

lab<-read.csv(file.choose())
View(lab)

attach(lab)

shapiro.test(Laboratory.1)
qqnorm(Laboratory.1)
qqline(Laboratory.1)

shapiro.test(Laboratory.2)
qqnorm(Laboratory.2)
qqline(Laboratory.2)

shapiro.test(Laboratory.3)
qqnorm(Laboratory.3)
qqline(Laboratory.3)

shapiro.test(Laboratory.4)
qqnorm(Laboratory.4)
qqline(Laboratory.4)

lab_stack<-stack(lab)
View(lab_stack)

names(lab_stack)[2]<-"Lab_number"

var_test<-bartlett.test(values~Lab_number,data=lab_stack)
var_test

aov_res<-aov(values~Lab_number,data=lab_stack)
summary(aov_res)

################   Chi Square test   ###################

buyer<-read.csv(file.choose())
View(buyer)

chisq.test(buyer[,-1])

#########################################################

order<-read.csv(file.choose())
View(order)

order1<-data.frame(order)
order1<-ifelse(order1=="Defective",1,0)
View(order1)

?stack
order1<-as.data.frame(order1)
order2<-stack(order1)
View(order2)
names(order2)<-c("Defective","Country")

contigency_table<-table(order2$Country,order2$Defective)
chisq.test(contigency_table)


############  2 proportion test   ###############

fantaloon<-read.csv(file.choose())
View(fantaloon)

fantaloon<-ifelse(fantaloon=="Male",1,0)
fantaloon<-stack(as.data.frame(fantaloon))
names(fantaloon)<-c("Gender","Day")

table(fantaloon$Day,fantaloon$Gender)
prop.test(table(fantaloon$Day,fantaloon$Gender))
