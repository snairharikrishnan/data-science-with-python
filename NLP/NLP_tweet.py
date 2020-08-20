import pandas as pd
import tweepy 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import seaborn as sns 

consumer_key = "Kq4mCtnOSPiNwA9ArvYq03DE7"
consumer_secret = "aWBfVbrJWppmEy3mAbrjUHa6Y8AKU6qkCBZwA6ZpAO8BEFaoC2"
access_key = "529590041-eZXHHkluorWkdRZRWiVYW3GVBuvr3VXt84cZcDYA"
access_secret = "rqlG8jzmKTPU3bZoCwgRnOUoD5UYOx8KDjhoXySPrR3mI"

alltweets = []	

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(access_key, access_secret)
api = tweepy.API(auth)    

new_tweets=api.user_timeline(screen_name="narendramodi",count=200)
len(new_tweets)
alltweets.extend(new_tweets)
alltweets[0].text
tweet_count=len(alltweets)

alltweets=[alltweets[i] for i in range(len(alltweets)) if alltweets[i].lang =='en'] #removing non english tweets
len(alltweets)

tweet_text=[alltweets[i].text for i in range(len(alltweets))]
tweet_text[0]

sia=SentimentIntensityAnalyzer()
def get_sentiment(tweet):
    analysis=sia.polarity_scores(tweet)
    if analysis['compound'] > 0:
        return 'positive'
    elif analysis['compound'] < 0:
        return 'negative'
    else:
        return 'neutral'


tweet_sentiment_table=pd.DataFrame(columns=("Tweet_ID","Tweet","Retweet_Count","Date","Sentiment"))
for i in range(len(alltweets)):
    tweet_sentiment_table=tweet_sentiment_table.append({"Tweet_ID":alltweets[i].id,
                                                        "Tweet":alltweets[i].text, 
                                                        "Retweet_Count":alltweets[i].retweet_count,
                                                        "Date":alltweets[i].created_at,
                                                        "Sentiment":get_sentiment(alltweets[i].text)},ignore_index=True)

print(tweet_sentiment_table.Sentiment.value_counts())
sns.countplot(tweet_sentiment_table.Sentiment)
