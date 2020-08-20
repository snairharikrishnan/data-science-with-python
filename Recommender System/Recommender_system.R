movies<-read.csv(file.choose())
View(movies)
str(movies)

install.packages("recommenderlab",dependencies = TRUE)
library("recommenderlab")
hist(movies[,3])

movies_matrix<-as(movies, "realRatingMatrix")

movies_rec<-Recommender(movies_matrix,method="POPULAR")

recommended_movies<-predict(movies_rec,movies_matrix[400],n=5)
as(recommended_movies,"list")

movies_rec1<-Recommender(movies_matrix,method="UBCF")
recommended_movies<-predict(movies_rec1,movies_matrix[775:800],n=5)
as(recommended_movies,"list")

movies_rec2<-Recommender(movies_matrix,method="IBCF")
recommended_movies<-predict(movies_rec2,movies_matrix[c(775,800)],n=5)
as(recommended_movies,"list")

# User-based collborative filtering (UBCF)
# Item-based collborative filtering (IBCF)
# SVD with column-mean imputation (SVD)
# Funk SVD (SVDF)
# Alternating Least Squares (ALS)
# MAtrix factorization with LIBMF (LIBMF)
# Association rule-based recommender (AR)
# Popular items (POPULAR)
# Randomly chosen items for comparison (RANDOM)
# Re-recommend liked items (RERECOMMEND)
# Hybrid recommendations (HybridRecommender)
