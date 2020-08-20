book<-read.csv(file.choose())
View(book)

library("arules")
library("arulesViz")

rules_book<-apriori(as.matrix(book),parameter = list(support=0.01,confidence=0.5,minlen=2))
inspect(rules_book)

rules_sorted<-sort(rules_book,by="lift")
inspect(rules_sorted[1:10])
# people who buy {RefBks,GeogBks,ItalArt} are highly likely to buy {ItalAtlas}  lift ratio=23.02 confidence=0.85 given support>0.01

rules_book<-apriori(as.matrix(book),parameter = list(support=0.02,confidence=0.5,minlen=2))
rules_sorted<-sort(rules_book,by="lift")
inspect(rules_sorted[1:10])
# people who buy {DoItYBks,ArtBks,ItalCook} are highly likely to buy {ItalArt}  lift ratio=14.12 confidence=0.68 given support>0.02

rules_book<-apriori(as.matrix(book),parameter = list(support=0.03,confidence=0.75,minlen=3))
rules_sorted<-sort(rules_book,by="lift")
inspect(rules_sorted[1:10])
# people who buy {CookBks,ItalArt} are highly likely to buy {ItalCook}  lift ratio=8.05 confidence=0.91 given support>0.03

plot(rules_sorted)
plot(rules_sorted,method = "grouped")
plot(rules_sorted,method = "graph")

#############################################################

movies<-read.csv(file.choose())
View(movies)

movies<-movies[,-c(1:5)]

rules_movies<-apriori(as.matrix(movies),parameter = list(support=0.1,confidence=0.5,minlen=2))
inspect(rules_movies)

rules_sorted<-sort(rules_movies,by="lift")
inspect(rules_sorted[1:5])
# people who see {Gladiator,Green.Mile} are highly likely to see {LOTR}  lift ratio=10 confidence=1 given support>0.1

rules_movies<-apriori(as.matrix(movies),parameter = list(support=0.2,confidence=0.75,minlen=3))
rules_sorted<-sort(rules_movies,by="lift")
inspect(rules_sorted)
# people who see {Sixth.Sense,Patriot} are highly likely to see {Gladiator}  lift ratio=1.42 confidence=1 given support>0.2

rules_movies<-apriori(as.matrix(movies),parameter = list(support=0.1,confidence=1,minlen=4))
rules_sorted<-sort(rules_movies,by="lift")
inspect(rules_sorted[1:5])
# people who see {Sixth.Sense,Gladiator,Green.Mile} are highly likely to see {LOTR}  lift ratio=10 confidence=1 given support>0.1

plot(rules_sorted)
plot(rules_sorted,method = "grouped")
plot(rules_sorted,method = "graph")

################################################################

groceries<-read.csv(file.choose())
View(groceries)

groceries[]<-lapply(groceries,as.character)

paste_fun <- function(i){
  return (paste(as.character(i),collapse=" "))
}
groceries["combination"] <- apply(groceries,1,paste_fun)

library(tm)
x <- Corpus(VectorSource(groceries$combination))
x <- tm_map(x,stripWhitespace)
dtm0 <- t(TermDocumentMatrix(x))
groceries_dtm <- data.frame(as.matrix(dtm0))
View(groceries_dtm)

detach(package:tm, unload = TRUE)
library("arules")
library("arulesViz")

rules_groceries<-apriori(as.matrix(groceries_dtm),parameter= list(support=0.01,confidence=0.5,minlen=2))
inspect(rules_groceries)
rules_sorted<-sort(rules_groceries,by="lift")
inspect(rules_sorted[1:10])
# people who buy {plants} are highly likely to buy {pot}  lift ratio=89.97 confidence=1 given support>0.01

rules_groceries<-apriori(as.matrix(groceries_dtm),parameter = list(support = 0.02,confidence = 1,minlen=2))
rules_sorted<-sort(rules_groceries,by="lift")
inspect(rules_sorted[1:10])
# people who buy {articles} are highly likely to buy {hygiene}  lift ratio=47.20 confidence=1 given support>0.02

rules_groceries<-apriori(as.matrix(groceries_dtm),parameter = list(support = 0.02,confidence = 1,minlen=3))
rules_sorted<-sort(rules_groceries,by="lift")
inspect(rules_sorted[1:10])
# people who buy {long,product} are highly likely to buy {life}  lift ratio=41.56 confidence=1 given support>0.02

plot(rules_sorted)
plot(rules_sorted,method = "grouped")
plot(rules_sorted,method = "graph")

#########################################################