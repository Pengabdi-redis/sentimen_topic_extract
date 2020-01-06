import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re, nltk
from nltk.tokenize import sent_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.base import BaseEstimator, TransformerMixin
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer ,TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pickle

factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

capres = pd.read_csv('data/Capres2014-2.0.csv',encoding='latin1')
capres.head()

def handle_emojis(tweet):
    tweet = re.sub(r'(:\s?\)|:-\)|\(\s?:|\(-:|:\'\))', ' EMO_POS ', tweet)
    tweet = re.sub(r'(:\s?D|:\s?p|:-D|x-?D|X-?D)', ' EMO_POS ', tweet)
    tweet = re.sub(r'(<3|:\*)', ' EMO_POS ', tweet)
    tweet = re.sub(r'(;-?\)|;-?D|\(-?;)', ' EMO_POS ', tweet)
    tweet = re.sub(r'(:\s?\(|:-\(|\)\s?:|\)-:)', ' EMO_NEG ', tweet)
    tweet = re.sub(r'(:,\(|:\'\(|:"\()', ' EMO_NEG ', tweet)
    return tweet

def normalizer(tweet): 
    tweet = handle_emojis(tweet)
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub(r'@[\S]+', 'USER_MENTION', tweet)
    tweet = tweet.replace("USER_MENTION", "")
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)    
    tweet = tweet.strip('\'"')
    tweet = re.sub("[^a-zA-Z]", " ",tweet)
    rpt_regex = re.compile(r"(.)\1{1,}", re.IGNORECASE)
    tweet = rpt_regex.sub(r"\1\1", tweet)
    stop = stopword.remove(tweet)
    return stop

capres['normalized_tweet']=capres.Isi_Tweet.apply(lambda x: normalizer(x))

feature = capres.normalized_tweet
label = capres.Sentimen

print(f'Shape of X array: \n{feature.shape}')
print(f'\nShape of y array: \n{label.shape}')


X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.3)

print(f'Shape of X train: \n{X_train.shape}')
print(f'\nShape of X test: \n{X_test.shape}')

pipeline= Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf',TfidfTransformer()),
    ('classifier',SVC(kernel = 'linear', random_state = 0))
])

print(type(X_train))

pipeline.fit(X_train,y_train)

predictions = pipeline.predict(X_test)

print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))

print(pipeline.predict(['jangan pilih koruptor']))

print(pipeline.predict(['siapapun presidennya tetep cari uang sendiri']))

print(pipeline.predict(['bodo amat']))

with open("model.pkl","wb") as model_file:
    pickle.dump(pipeline, model_file)




