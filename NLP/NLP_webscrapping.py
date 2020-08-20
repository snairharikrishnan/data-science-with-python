from bs4 import BeautifulSoup
import requests

response=requests.get("https://www.google.com/")
print(response.status_code)  #200 if ok
print(response.headers)  # to verify the page

src=response.content

soup=BeautifulSoup(src,'lxml')

links=soup.find_all("a") # all the links on the page <a> tag for hyperlink 
print(links)

for link in links:
    if "About" in link.text:        # to get the link that has About as text in it
        print(link)
        print(link.text)
        print(link.attrs['href'])
        
#######################################################################
from bs4 import BeautifulSoup
import requests
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

response=requests.get("https://www.amazon.in/All-New-Kindle-reader-Glare-Free-Touchscreen/product-reviews/B0186FF45G/ref=cm_cr_getr_d_paging_btm_3?showViewpoints=1&pageNumber=")
print(response.status_code)

soup=BeautifulSoup(response.content,'html.parser')
reviews=soup.find_all("span",attrs={"class","a-size-base review-text review-text-content"})
len(reviews) #10 reviews on the page
reviews[0].text

kindle_reviews=[]
for i in range(len(reviews)):
    kindle_reviews.append(reviews[i].text)

all_reviews=" ".join(kindle_reviews) # joining all reviews

all_reviews=re.sub('[^A-Za-z' ']+',' ',all_reviews)  # only a single space will be present
all_reviews=all_reviews.lower()

review_words=all_reviews.split(" ")
len(review_words)

stop_words=stopwords.words('english')
review_words=[word for word in review_words if word not in stop_words]
len(review_words)

all_reviews=" ".join(review_words)

wc=WordCloud(width=1800,height=1400,background_color='black').generate(all_reviews)
plt.rcParams.update({'figure.figsize':(8,6),'figure.dpi':120})
plt.imshow(wc)

with open("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\positive-words.txt","r") as pos:
    positive_words=pos.read().split("\n")

with open("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\negative-words.txt","r") as neg:
    negative_words=neg.read().split("\n")

pos_words_review=" ".join([word for word in review_words if word in positive_words])
neg_words_review=" ".join([word for word in review_words if word in negative_words])

wc_pos=WordCloud(width=1800,height=1400,background_color='black').generate(pos_words_review)
plt.imshow(wc_pos)

wc_neg=WordCloud(width=1800,height=1400,background_color='black').generate(neg_words_review)
plt.imshow(wc_neg,interpolation="bilinear")

#LDA Webinar

########################################################################

# Sentiment Analysis
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def get_sentiment(review):
    analysis=TextBlob(review)
    if analysis.sentiment.polarity>0:
        return 'positive'
    elif analysis.sentiment.polarity==0:
        return 'neutral'
    else:
        return 'negative'


def get_vader_sentiment(review):
    sia=SentimentIntensityAnalyzer()
    analysis=sia.polarity_scores(review)
    if analysis['compound'] > 0:
        return 'positive'
    elif analysis['compound'] <0:
        return 'negative'
    else:
        return 'neutral'


url="https://www.imdb.com/title/tt0903747/reviews?ref_=tt_ql_3"
response=requests.get(url)
response.status_code

soup=BeautifulSoup(response.content,'html.parser')
reviews=soup.find_all(attrs={"class","text show-more__control"})
len(reviews)

breaking_bad_reviews=[]
for r in reviews:
    breaking_bad_reviews.append(r.text)

sentiment_table=pd.DataFrame(columns=("Review_Number","Sentiment"))
for i in range(len(breaking_bad_reviews)):
    sentiment_table = sentiment_table.append({"Review_Number": i,"Sentiment":get_vader_sentiment(breaking_bad_reviews[i])},ignore_index=True)
    

