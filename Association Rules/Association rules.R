install.packages("arules")
install.packages("arulesViz")
library("arules")
library("arulesViz")

phone<-read.csv(file.choose())
View(phone)
phone<-phone[,4:9]

rules=apriori(as.matrix(phone),parameter = list(support=0.02,confidence=0.5,minlen=3))
inspect(rules)

rules.sorted=sort(rules,by="lift")
inspect(rules.sorted[1:5])
plot(rules.sorted)

plot(rules,method = "scatterplot")
plot(rules,method = "grouped")
plot(rules,method = "graph")

book<-read.csv(file.choose())
View(book)

rules<-apriori(as.matrix(book),parameter = list(support=0.01,confidence=0.5,minlen=3))
inspect(rules)

rules.sorted<-sort(rules,by="lift")
inspect(rules.sorted[1:20])
