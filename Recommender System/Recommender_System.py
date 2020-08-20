import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.metrics.pairwise import linear_kernel
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

book = pd.read_excel("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\Book_recommender.xlsx",usecols=['User.ID','Book.Title','Book.Rating'])
book.rename(columns={"User.ID":"User_ID","Book.Title":"Book_Title","Book.Rating":"Book_Rating"},inplace=True)
book['User_ID'].value_counts()
book['Book_Title'].value_counts()

for i in range(10000):
    book['Book_Title'][i]=str(book['Book_Title'][i])
    
#Content Based Recommendation
tfidf=TfidfVectorizer(stop_words='english')

book['Book_Title'].isnull().sum()
type(book['Book_Title'][0])

for i in range(len(book)):
    book['Book_Title'][i]=re.sub('[^A-Za-z0-9' ']+',' ',str(book['Book_Title'][i]))
    
title_vector=tfidf.fit_transform(book['Book_Title'])
title_vector.shape

lin=linear_kernel(title_vector,title_vector)
lin[150]
indices=pd.Series(book.index,index=book['Book_Title']).drop_duplicates()

def get_recommendation(Name,num):
    idx=indices[Name]
    lin_scores=list(enumerate(lin[idx]))
    lin_scores=sorted(lin_scores,key=lambda x:x[1],reverse=True)
    top_n=lin_scores[1:num+1]
    top_scores=[i[1] for i in top_n]
    top_index=[i[0] for i in top_n]
    
    recommendation=pd.DataFrame(columns=("Name","Score"))
    recommendation.Name=book.loc[top_index,"Book_Title"]
    recommendation.Score=top_scores
    recommendation.reset_index(inplace=True)
    recommendation.drop(['index'],axis=1,inplace=True)
    print(recommendation)
    
    
get_recommendation("The Invasion Animorphs No 1 ",15)
#Name="The Invasion Animorphs No 1 ";num=15

# Collaborative Filtering
#restart kernel and run upto line 14
count=book.groupby(by=['Book_Title'])['Book_Rating'].count().reset_index().rename(columns={'Book_Rating':'Total_Rating'})[['Book_Title','Total_Rating']]
count.sort_values(['Total_Rating'],ascending=False,inplace=True)
new_book=book.merge(count,left_on='Book_Title',right_on='Book_Title',how='left')

#Max 5 people rated for each book
#Hence ignoring count threshold

book_pivot=new_book.pivot_table(index='Book_Title',columns='User_ID',values='Book_Rating').fillna(0)
book_pivot.shape

book_matrix=csr_matrix(book_pivot.values)

knn_model=NearestNeighbors(metric='cosine',algorithm='brute')
knn_model.fit(book_matrix)

def get_recommendations(Name,num):
    loc=0
#    Name="Icebound"
    for i in range(9659):                         #for validation
        p=pd.DataFrame(book_pivot.iloc[i,:])
        if p.columns[0]==Name:
            loc=i
            break
    
    dist,index=knn_model.kneighbors(book_pivot.loc[Name,:].values.reshape(1,-1),n_neighbors=num+1)
    recommendation=pd.DataFrame(columns=("Name","Distance"))
    for i in range(len(dist.flatten())):
        if i==0:
            print(f"\nRecommendations for {book_pivot.index[loc]}\n")
        else:
            recommendation=recommendation.append({'Name':book_pivot.index[index.flatten()[i]],
                                                  'Distance':dist.flatten()[i]},ignore_index=True)
    
    print(recommendation)

#a=[];b=[]
#for i in range(9659):
#    Name=book['Book_Title'][i]
#    dist,index=knn_model.kneighbors(book_pivot.iloc[i,:].values.reshape(1,-1),n_neighbors=5)
#    x=[i for i in dist.flatten() if i !=0]
#    if len(x) !=0:
#        a.append(Name)
#        b.append(i)
       
get_recommendations("The Testament",6)   
