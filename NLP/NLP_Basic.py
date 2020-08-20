import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

paragraph="""Born and raised in a Hindu family in coastal Gujarat, western India, 
            Gandhi was trained in law at the Inner Temple, London, and called to the bar at age 22 in June 1891. 
            After two uncertain years in India, where he was unable to start a successful law practice, he moved 
            to South Africa in 1893 to represent an Indian merchant in a lawsuit. He went on to stay for 21 years. 
            It was in South Africa that Gandhi raised a family, and first employed nonviolent resistance in a campaign 
            for civil rights. In 1915, aged 45, he returned to India. He set about organising peasants, farmers, and 
            urban labourers to protest against excessive land-tax and discrimination. Assuming leadership of the 
            Indian National Congress in 1921, Gandhi led nationwide campaigns for easing poverty, expanding women's rights, 
            building religious and ethnic amity, ending untouchability, and above all for achieving Swaraj or self-rule"""
            
sentences=nltk.sent_tokenize(paragraph) #list of sentences

stemmer=PorterStemmer()
lemmatizer=WordNetLemmatizer()

#stemming
for i in range(len(sentences)):
    words=nltk.word_tokenize(sentences[i])     #list of words
    words=[stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i]=" ".join(words)
    

#lemmetizing
sentences=nltk.sent_tokenize(paragraph) #list of sentences
for i in range(len(sentences)):
    words=nltk.word_tokenize(sentences[i])     #list of words
    words=[lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i]=" ".join(words)
    
#Cleaning data ie removing numbers and special characters
import re
corpus=[]
sentences=nltk.sent_tokenize(paragraph) #list of sentences
for i in range(len(sentences)):
    words=re.sub('[^a-zA-Z]',' ',sentences[i])
    words=words.lower()
    words=words.split()
    words=[lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    words=" ".join(words)
    corpus.append(words)
    
#Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()    
vector=cv.fit_transform(corpus).toarray()

#TFIDF
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer()
vector=tfidf.fit_transform(corpus).toarray()



#########################################################################
#Word2Vec
from gensim.models import Word2Vec
corpus=[]
sentences=nltk.sent_tokenize(paragraph) #list of sentences
for i in range(len(sentences)):
    words=re.sub('[^a-zA-Z]',' ',sentences[i])
    words=words.lower()
    words=words.split()
    words=[lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    words=" ".join(words)
    corpus.append(words)

corpus=[nltk.word_tokenize(word) for word in corpus]

model=Word2Vec(corpus,min_count=1)  #ignore words whose count is less than 1
words=model.wv.vocab

#finding word vector
vec=model.wv['civil']

#finding similaar words
similar=model.wv.most_similar('gandhi')
