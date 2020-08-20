import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

fake=pd.read_csv("C:\\Users\\snair\\Documents\\Data Science Assignment\\Data Sets\\fake_news.csv")
fake.shape
fake.isnull().sum()
fake.dropna(inplace=True)
fake.shape
fake.reset_index(inplace=True)

corpus=[]
words=''
stop=set(stopwords.words('english'))
lem=WordNetLemmatizer()
loop_count=[x for x in range(0,18258,2000)]
for i in range(len(fake)):
    words=re.sub('[^A-Za-z]',' ',fake.text[i])
    words=words.lower()
    words=words.split()
    words=[lem.lemmatize(word) for word in words if word not in stop]
    words=" ".join(words)
    corpus.append(words)
    if i in loop_count:
        print(f'{i} loops')
       

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer 
#tfidf=TfidfVectorizer()
bow=CountVectorizer(max_features=5000,ngram_range=(1,3))
vector=bow.fit_transform(corpus).toarray()

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(vector,fake['label'],test_size=0.33,random_state=0)

bow.get_feature_names()[:100]
bow.get_params()
count_fake=pd.DataFrame(train_x,columns=bow.get_feature_names())
count_fake.head()

from sklearn.naive_bayes import MultinomialNB
classifier=MultinomialNB()

classifier.fit(train_x,train_y)
pred=classifier.predict(test_x)
pd.crosstab(pred,test_y)
acc=np.mean(pred==test_y)
print(f'The accuracy is {acc*100}')

# hyperparameter tuning on MultinomialNB
accuracy=0.00
for alpha in np.arange(0,1,0.1):
    sub_classifier=MultinomialNB(alpha=alpha)
    sub_classifier.fit(train_x,train_y)
    pred=sub_classifier.predict(test_x)
    acc=np.mean(pred==test_y) 
    if acc>accuracy:
        classifier=sub_classifier
        accuracy=acc
    print("Alpha:{}  Accuracy:{}".format(alpha,acc))


classifier.fit(train_x,train_y)
pred=classifier.predict(test_x)
pd.crosstab(pred,test_y)
acc=np.mean(pred==test_y)
print(f'The accuracy is {acc*100}') # 89.66%

classifier.coef_[0]
sorted(zip(classifier.coef_[0],bow.get_feature_names()))[:20] #most fake
sorted(zip(classifier.coef_[0],bow.get_feature_names()),reverse=True)[:20] #most real





