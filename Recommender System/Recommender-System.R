book<-read.csv(file.choose())
View(book)

hist(book$Book.Rating)

install.packages("recommenderlab",dependencies = TRUE)
library("recommenderlab")

book_matrix<-as(book[,-1],"realRatingMatrix")

book_rec<-Recommender(book_matrix,method="POPULAR")
recommended_books<-predict(book_rec,book_matrix[1000],n=5)
as(recommended_books,"list")

book_rec1<-Recommender(book_matrix,method="UBCF")
recommended_books<-predict(book_rec1,book_matrix[150:160],n=5)
as(recommended_books,"list")

# book_rec2<-Recommender(book_matrix,method="IBCF")     data is based on rating
# recommended_books<-predict(book_rec2,book_matrix[150:160],n=5)
# as(recommended_books,"list")
