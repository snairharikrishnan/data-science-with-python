import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import requests
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud

url=["https://www.flipkart.com/apple-iphone-se-black-64-gb/product-reviews/itm832dd5963a08d?pid=MOBFRFXHCKWDAC4A&aid=overall&certifiedBuyer=false&sortOrder=NEGATIVE_FIRST&page=1",
     "https://www.flipkart.com/apple-iphone-se-black-64-gb/product-reviews/itm832dd5963a08d?pid=MOBFRFXHCKWDAC4A&aid=overall&certifiedBuyer=false&sortOrder=POSITIVE_FIRST",
     "https://www.flipkart.com/apple-iphone-se-black-64-gb/product-reviews/itm832dd5963a08d?pid=MOBFRFXHCKWDAC4A&aid=overall&certifiedBuyer=false&sortOrder=NEGATIVE_FIRST&page=2",
     "https://www.flipkart.com/apple-iphone-se-black-64-gb/product-reviews/itm832dd5963a08d?pid=MOBFRFXHCKWDAC4A&aid=overall&certifiedBuyer=false&sortOrder=MOST_HELPFUL",
     "https://www.flipkart.com/apple-iphone-se-black-64-gb/product-reviews/itm832dd5963a08d?pid=MOBFRFXHCKWDAC4A&aid=overall&certifiedBuyer=false&sortOrder=MOST_RECENT",
     "https://www.flipkart.com/apple-iphone-se-black-64-gb/product-reviews/itm832dd5963a08d?pid=MOBFRFXHCKWDAC4A&aid=overall&certifiedBuyer=false&sortOrder=POSITIVE_FIRST&page=2"]

all_reviews=[]
for i in range(len(url)):
    response=requests.get(url[i])
    print(f"Status Code= {response.status_code}") 
    soup=BeautifulSoup(response.content,'html.parser')
    reviews=soup.find_all(attrs={"class","qwjRop"})
    print(f"Reviews Extracted {len(reviews)}")
    all_reviews.append(reviews)
    
all_reviews_text=[] 
for i in range(len(all_reviews)):
    for j in range(len(reviews)):
        all_reviews_text.append(all_reviews[i][j].text)


sia=SentimentIntensityAnalyzer()
def get_sentiment(review):
    analysis=sia.polarity_scores(review)
    if analysis['compound']>0:
        return 'positive'        
    elif analysis['compound']<0:
        return 'negative'        
    else:
        return 'neutral'


review_table=pd.DataFrame(columns=("Review_number","Review","Sentiment"))

for i in range(len(all_reviews_text)):
    review_table=review_table.append({'Review_number':i+1,
                                      'Review':all_reviews_text[i],
                                      'Sentiment':get_sentiment(all_reviews_text[i])},ignore_index=True)

print(review_table)
sns.countplot(review_table.Sentiment) #more positive comments
pos_percent=review_table.Sentiment.value_counts()[0]*100/len(all_reviews_text)
neg_percent=review_table.Sentiment.value_counts()[1]*100/len(all_reviews_text)
neutral_percent=review_table.Sentiment.value_counts()[2]*100/len(all_reviews_text)

print("{} percent positive reviews\n{} percent negative reviews\n{} percent neutral reviews".format(pos_percent,neg_percent,neutral_percent))

    
lemmetizer=WordNetLemmatizer()
all_reviews_text=" ".join(all_reviews_text)
all_reviews_text=re.sub('[^A-Za-z' ']+',' ',all_reviews_text)
all_reviews_text=all_reviews_text.lower()  
all_reviews_text=all_reviews_text.split()  
all_reviews_text=[lemmetizer.lemmatize(word) for word in all_reviews_text if word not in set(stopwords.words('english'))]

pos_words=[]
neg_words=[]
for word in all_reviews_text:
    if get_sentiment(word)=='positive':
        pos_words.append(word)
    elif get_sentiment(word)=='negative':
        neg_words.append(word)

len(pos_words)
len(neg_words)

all_reviews_text=" ".join(all_reviews_text)    
plt.rcParams.update({'figure.figsize':(8,6),'figure.dpi':120})
wc=WordCloud(width=1800,height=1400,background_color='black').generate(all_reviews_text)
plt.imshow(wc,interpolation="bilinear") #Word Cloud


pos_words=" ".join(pos_words)
wc_pos=WordCloud(width=1800,height=1400,background_color='black').generate(pos_words)
plt.imshow(wc_pos,interpolation='bilinear') #positive Word Cloud

neg_words=" ".join(neg_words)
wc_neg=WordCloud(width=1800,height=1400,background_color='black').generate(neg_words)
plt.imshow(wc_neg,interpolation='bilinear') #negative Word Cloud

##############################################################################

url="https://www.imdb.com/title/tt0944947/reviews?ref_=tt_ql_3"
    
response=requests.get(url)
print(f'Status Code= {response.status_code}')
soup=BeautifulSoup(response.content,'html.parser')
reviews=soup.find_all(attrs={"class":"text show-more__control"})
print(f'{len(reviews)} reviews downloaded')

got_reviews=[]
for r in reviews:
    got_reviews.append(r.text)


sia=SentimentIntensityAnalyzer()
def get_sentiment(review):
    analysis=sia.polarity_scores(review)
    if analysis['compound']>0:
        return 'positive'        
    elif analysis['compound']<0:
        return 'negative'        
    else:
        return 'neutral'


review_table=pd.DataFrame(columns=("Review_number","Review","Sentiment"))

for i in range(len(got_reviews)):
    review_table=review_table.append({'Review_number':i+1,
                                      'Review':got_reviews[i],
                                      'Sentiment':get_sentiment(got_reviews[i])},ignore_index=True)

print(review_table)
sns.countplot(review_table.Sentiment)
pos_percent=review_table.Sentiment.value_counts()[0]*100/len(got_reviews)
neg_percent=review_table.Sentiment.value_counts()[1]*100/len(got_reviews)

print("{} percent positive reviews\n{} percent negative reviews".format(pos_percent,neg_percent))


lemmetizer=WordNetLemmatizer()
got_reviews=" ".join(got_reviews)
got_reviews=re.sub('[^A-Za-z' ']+',' ',got_reviews)
got_reviews=got_reviews.lower()  
got_reviews=got_reviews.split()  
got_reviews=[lemmetizer.lemmatize(word) for word in got_reviews if word not in set(stopwords.words('english'))]

pos_words=[]
neg_words=[]
for word in got_reviews:
    if get_sentiment(word)=='positive':
        pos_words.append(word)
    elif get_sentiment(word)=='negative':
        neg_words.append(word)

len(pos_words)
len(neg_words)

got_reviews=" ".join(got_reviews)    
wc=WordCloud(width=1800,height=1400,background_color='black').generate(got_reviews)
plt.imshow(wc) 


pos_words=" ".join(pos_words)
wc_pos=WordCloud(width=1800,height=1400,background_color='black').generate(pos_words)
plt.imshow(wc_pos) #positive Word Cloud

neg_words=" ".join(neg_words)
wc_neg=WordCloud(width=1800,height=1400,background_color='black').generate(neg_words)
plt.imshow(wc_neg) #negative Word Cloud




