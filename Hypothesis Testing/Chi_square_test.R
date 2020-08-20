Bahaman<-read.csv(file.choose())
View(Bahaman)
attach(Bahaman)

table(Defective,Country)
table(Country,Defective)

t1<-prop.table(table(Defective))   # to see % of defects
t1

?chisq.test
chisq.test(table(Country,Defective))
#p high null fly
#all countries having equal proportion