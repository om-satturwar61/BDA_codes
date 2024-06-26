# -*- coding: utf-8 -*-
"""BDA_Assi_3_twitter_sentiment_analysis.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MstyMUGxP61XnKSPBgB6DdMFu8vrWzHz
"""

import pandas as pd
from textblob import TextBlob

df = pd.read_csv('twitter_training.csv')
df

df.columns=['No','Category','Sentiment','text']
df

print(train[train['label'] == 0].head())
print('--------------------------------------')
print(train[train['label'] == 1].head())

def classify_tweet(tweet):
    analysis = TextBlob(tweet)
    # Assuming you consider negative sentiment as 'hate'
    return 'hate' if analysis.sentiment.polarity < 0 else 'not hate'

df['sentiment'] = df['text'].apply(lambda x: classify_tweet(x))



df['sentiment'] = df['text'].apply(classify_tweet)

# Commented out IPython magic to ensure Python compatibility.
import re    # for regular expressions
import nltk  # for text manipulation
import string
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", 200)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# %matplotlib inline

import pandas as pd

train  = pd.read_csv('train_E6oV3lV.csv')
test = pd.read_csv('test_tweets_anuFYb8.csv')

print(train[train['label'] == 0].head())
print('--------------------------------------')
print(train[train['label'] == 1].head())

train[train['label'] == 0].head(10)

train[train['label'] == 1].head(10)

train.shape

test.shape

length_train = train['tweet'].str.len()
length_test = test['tweet'].str.len()
plt.hist(length_train, bins=20, label="train_tweets")
plt.hist(length_test, bins=20, label="test_tweets")
plt.legend()
plt.show()

combi = train.append(test, ignore_index=True)
combi.shape

def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt

combi['tidy_tweet'] = np.vectorize(remove_pattern)(combi['tweet'], "@[w]*")
combi.head()

combi['tidy_tweet'] = combi['tidy_tweet'].str.replace("[^a-zA-Z#]", " ")
combi.head(10)

combi['tidy_tweet'] = combi['tidy_tweet'].apply(lambda x: ' '.join([w
                        for w in x.split() if len(w)>3]))

tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing
tokenized_tweet.head()

from nltk.stem.porter import *
stemmer = PorterStemmer()
tokenized_tweet = tokenized_tweet.apply(lambda x: [stemmer.stem(i) for i in x])

for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = ' '.join(tokenized_tweet[i])
combi['tidy_tweet'] = tokenized_tweet

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import gensim

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(combi['tidy_tweet'])
bow.shape

tfidf_vectorizer = TfidfVectorizer(max_df=0.90, min_df=2, max_features=1000,stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(combi['tidy_tweet'])
tfidf.shape

tokenized_tweet = combi['tidy_tweet'].apply(lambda x: x.split()) # tokenizing

model_w2v = gensim.models.Word2Vec(tokenized_tweet, vector_size=200, window=5,min_count = 2,sg = 1,hs= 0,negative = 10,workers =2,seed= 34)

model_w2v.train(tokenized_tweet, total_examples= len(combi['tidy_tweet']), epochs=20)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# Extracting train and test BoW features
train_bow = bow[:31962,:]
test_bow = bow[31962:,:]
# splitting data into training and validation set
xtrain_bow, xvalid_bow, ytrain, yvalid = train_test_split(train_bow, train['label'],
                                            random_state=42,test_size=0.3)
lreg = LogisticRegression()
# training the model
lreg.fit(xtrain_bow, ytrain)
prediction = lreg.predict_proba(xvalid_bow) # predicting on the validation set
prediction_int = prediction[:,1] >= 0.3 # if prediction is greater than or equal to 0.3 than 1 else 0
prediction_int = prediction_int.astype(np.int)
f1_score(yvalid, prediction_int) # calculating f1 score for the validation set

print(test['label'])

test_pred = lreg.predict_proba(test_bow)
test_pred_int = test_pred[:,1] >= 0.3
test_pred_int = test_pred_int.astype(np.int)
test['label'] = test_pred_int
submission = test[['id','label']]
submission.to_csv('sub_lreg_bow.csv', index=False) # writing data to a CSV file
true_labels = test['id']

f1 = f1_score(true_labels, test_pred_int)

print(test['id'])

from sklearn.metrics import classification_report

# Assuming 'submission' is your DataFrame with 'id' and 'label' columns
true_labels = submission['id']
predicted_labels = submission['label']

# Generate a classification report
report = classification_report(true_labels, predicted_labels)

# Print the classification report
print(report)