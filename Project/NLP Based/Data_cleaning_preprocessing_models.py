# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 13:22:16 2021

@author: anirudh.kumar.verma
"""

import json
import pandas as pd
import numpy as np
import nltk
import sklearn
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
import textblob
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import seaborn as sns


# Loading data
data = pd.read_json("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Project P58\\amazon_one_plus_reviews.json")
data.columns

# Data formating 
review_data = data.drop(['product_company', 'profile_name', 'review_title','helpful_count', 'total_comments',
       'review_country', 'reviewed_at', 'url', 'crawled_at', '_id',
       'verified_purchase', 'color', 'style_name', 'size_name', 'category',
       'sub_category', 'images'], axis=1)

review_data.drop(review_data.index[9001:],axis=0,inplace=True)
review_data.tail()


review_data['review_rating'].replace('5.0 out of 5 stars','5',inplace=True)
review_data['review_rating'].replace('4.0 out of 5 stars','4',inplace=True)
review_data['review_rating'].replace('3.0 out of 5 stars','3',inplace=True)
review_data['review_rating'].replace('2.0 out of 5 stars','2',inplace=True)
review_data['review_rating'].replace('1.0 out of 5 stars','1',inplace=True)

# EDA
review_data.info()
review_data.isnull().sum() # No null value

# Data visulaization
review_data['review_rating'].value_counts()
review_data.review_rating.value_counts().plot(kind="pie") # pie plot

sns.distplot(review_data['review_rating'])
sns.countplot(x='review_rating', data=review_data)

# Text Pre-processing

# Word count
review_data['word_count'] = review_data['review_text'].apply(lambda x: len(str(x).split(" ")))
review_data[['review_text','word_count']].head()

text = " ".join(review_data for review_data in review_data.review_text)

#Removing Punctuations 
review_data['review_text'] = review_data['review_text'].str.replace('[^\w\s]','')
review_data['review_text'].head()

#Removal of stop words
stop_words = set(stopwords.words('english'))
review_data['review_text'] = review_data['review_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))

#Normalize the data
review_data['review_text'] = review_data['review_text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
review_data['review_text'].head()

#Freq
freq = pd.Series(' '.join(review_data['review_text']).split()).value_counts()[:10]

freq

# removing high frequency words
freq = list(freq.index)
review_data['Text'] = review_data['review_text'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

# low frequency
low_freq = pd.Series(' '.join(review_data['Text']).split()).value_counts()[-1000:]
low_freq

#spelling correct
from textblob import TextBlob
review_data['Text'].apply(lambda x: str(TextBlob(x).correct()))

TextBlob(review_data['Text'][1]).words

# Lemmatization 
from textblob import Word
nltk.download('wordnet')
review_data['Text'] = review_data['Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()])) 

review_data['Text'].head()


reviews = review_data['Text']
reviews.dropna(inplace=True)

# Word cloud prep
text = " ".join(review for review in reviews)
print ("There are {} words in the combination of all review.".format(len(text)))

wordcloud = WordCloud(background_color='black').generate(text)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis("off")
plt.show()

#Calculating Sentiment score

import seaborn as sns
import re
import os
import sys
import ast
plt.style.use('fivethirtyeight')
# Function for getting the sentiment
cp = sns.color_palette()
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
emptyline=[]
for row in review_data['Text']:
    vs=analyzer.polarity_scores(row)
    emptyline.append(vs)
# Creating new dataframe with sentiments
df_sentiments=pd.DataFrame(emptyline)
df_sentiments.head()

#concat the scores with main data
review_data_sent=pd.concat([review_data.reset_index(drop=True), df_sentiments], axis=1)
review_data_sent.head()

#convert scores into positive and negetive using threshold
review_data_sent['Sentiment'] = np.where(review_data_sent['compound'] >= 0 , 'Positive','Negative')
review_data_sent.head(5)

# extract final data to csv
review_data_sent.to_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Project P58\\review_data.csv")

# Visualization on treated data
result=review_data_sent['Sentiment'].value_counts()
result.plot(kind='bar', rot=0, color=['plum','cyan'])

review_data_sent.Sentiment.value_counts().plot(kind="pie")

#Feature Extraction using CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df = 1, max_df = 0.9)
X = vectorizer.fit_transform(review_data_sent["Text"])
freq_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'occurrences':np.asarray(X.sum(axis=0)).ravel().tolist()})
freq_df['frequency'] = freq_df['occurrences']/np.sum(freq_df['occurrences'])

freq_df.head(5)



# TFidf vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features= 1000, max_df = 0.5, smooth_idf=True) #keep top 1000 words
doc_vec = vectorizer.fit_transform(review_data_sent["Text"])
names_features = vectorizer.get_feature_names()
dense = doc_vec.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns = names_features)


# Model Building

######## Logistic regression ############

review_data_sent["Text"].head()

from sklearn.model_selection import train_test_split
Independent_var = review_data_sent.Text
Dependent_var = review_data_sent.Sentiment

review_data_sent.Sentiment.value_counts()

IV_train,IV_test,DV_train,DV_test= train_test_split(Independent_var,Dependent_var, test_size=0.3, random_state=100)

print('IV_train:', len(IV_train))
print('IV test:', len(IV_test))
print('DV_train:', len(DV_train))
print('DV_test,', len(DV_test))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

tvec = TfidfVectorizer()
clf2 = LogisticRegression (solver = "lbfgs")

from sklearn.pipeline import Pipeline
model = Pipeline ([('vectorizer',tvec), ('classifier', clf2)])

model.fit(IV_train, DV_train)

from sklearn.metrics import confusion_matrix

predictions = model.predict(IV_test)

confusion_matrix(predictions, DV_test)

# Model Prediction

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

print("Accuracy: ", accuracy_score(predictions, DV_test))
print("Precision : ", precision_score (predictions, DV_test, average = 'weighted'))
print("Recall : ", recall_score (predictions, DV_test, average = 'weighted'))
print("F1_score : ", f1_score (predictions, DV_test, average = 'weighted'))

## prediction on all data

predictions_final = model.predict(Independent_var)

confusion_matrix(Dependent_var,predictions_final)

# Model Prediction

from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

print("Accuracy: ", accuracy_score(predictions_final, Dependent_var))
print("Precision : ", precision_score (predictions_final, Dependent_var, average = 'weighted'))
print("Recall : ", recall_score (predictions_final, Dependent_var, average = 'weighted'))
print("F1_score : ", f1_score (predictions_final, Dependent_var, average = 'weighted'))


#Trying on Example Review

example= ["I'm satisfied"] 
result = model.predict(example)

print (result)



###### Naive Bayes ####################

review_data_sent.head()

def split_into_words(i):
    return (i.split(" "))


from sklearn.model_selection import train_test_split

data_train,data_test = train_test_split(review_data_sent,test_size=0.3)

data_test

# Preparing data texts into word count matrix format using count vectorizer
data_bow = CountVectorizer(analyzer=split_into_words).fit(review_data_sent.Text)

# For all messages
all_datas_matrix = data_bow.transform(review_data_sent.Text)
all_datas_matrix.shape 

# For training messages
train_datas_matrix = data_bow.transform(data_train.Text)
train_datas_matrix.shape

# For testing messages
test_datas_matrix = data_bow.transform(data_test.Text)
test_datas_matrix.shape

# Preparing a naive bayes model on training Hamspam set 

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_datas_matrix,data_train.Sentiment)
train_pred_m = classifier_mb.predict(train_datas_matrix)
accuracy_train_m = np.mean(train_pred_m==data_train.Sentiment) # 98%

test_pred_m = classifier_mb.predict(test_datas_matrix)
accuracy_test_m = np.mean(test_pred_m==data_test.Sentiment) # 96%

all_pred_m = classifier_mb.predict(all_datas_matrix)
accuracy_all_m = np.mean(all_pred_m==review_data_sent.Sentiment) # 96%


from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score

review_data_sent.Sentiment.value_counts()
pd.crosstab(all_pred_m, review_data_sent.Sentiment)
print("Accuracy: ", accuracy_score(all_pred_m, review_data_sent.Sentiment))
print("Precision : ", precision_score (all_pred_m, review_data_sent.Sentiment, average = 'weighted'))
print("Recall : ", recall_score (all_pred_m, review_data_sent.Sentiment, average = 'weighted'))
print("F1_score : ", f1_score (all_pred_m, review_data_sent.Sentiment, average = 'weighted'))



# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(train_datas_matrix.toarray(),data_train.Sentiment.values) # we need to convert tfidf into array format which is compatible for gaussian naive bayes
train_pred_g = classifier_gb.predict(train_datas_matrix.toarray())
accuracy_train_g = np.mean(train_pred_g==data_train.Sentiment) # 95%

test_pred_g = classifier_gb.predict(test_datas_matrix.toarray())
accuracy_test_g = np.mean(test_pred_g==data_test.Sentiment) # 8%


all_pred_g = classifier_gb.predict(all_datas_matrix.toarray())
accuracy_all_g = np.mean(all_pred_g==review_data_sent.Sentiment) # 96%


pd.crosstab(all_pred_g, review_data_sent.Sentiment)
print("Accuracy: ", accuracy_score(all_pred_g, review_data_sent.Sentiment))
print("Precision : ", precision_score (all_pred_g, review_data_sent.Sentiment, average = 'weighted'))
print("Recall : ", recall_score (all_pred_g, review_data_sent.Sentiment, average = 'weighted'))
print("F1_score : ", f1_score (all_pred_g, review_data_sent.Sentiment, average = 'weighted'))


### Using TFIDF

from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(all_datas_matrix)

all_tfidf= tfidf_transformer.transform(all_datas_matrix)
# Preparing TFIDF for train datas
train_tfidf = tfidf_transformer.transform(train_datas_matrix)

train_tfidf.shape 

# Preparing TFIDF for test datas
test_tfidf = tfidf_transformer.transform(test_datas_matrix)

test_tfidf.shape 


# Preparing a naive bayes model on training set 

from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf,data_train.Sentiment)
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m==data_train.Sentiment) # 96%

test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m==data_test.Sentiment) # 96%

all_pred_m = classifier_mb.predict(all_tfidf)
accuracy_all_m = np.mean(all_pred_m==review_data_sent.Sentiment) # 96%


from sklearn.metrics import accuracy_score, precision_score, recall_score


pd.crosstab(all_pred_m, review_data_sent.Sentiment)
print("Accuracy: ", accuracy_score(all_pred_m, review_data_sent.Sentiment))
print("Precision : ", precision_score (all_pred_m, review_data_sent.Sentiment, average = 'weighted'))
print("Recall : ", recall_score (all_pred_m, review_data_sent.Sentiment, average = 'weighted'))
print("F1_score : ", f1_score (all_pred_m, review_data_sent.Sentiment, average = 'weighted'))

# Gaussian Naive Bayes 
classifier_gb = GB()
classifier_gb.fit(train_tfidf.toarray(),data_train.Sentiment.values) 
train_pred_g = classifier_gb.predict(train_tfidf.toarray())
accuracy_train_g = np.mean(train_pred_g==data_train.Sentiment)

test_pred_g = classifier_gb.predict(test_tfidf.toarray())
accuracy_test_g = np.mean(test_pred_g==data_test.Sentiment)

all_pred_g = classifier_gb.predict(all_tfidf.toarray())
accuracy_all_g = np.mean(all_pred_g==review_data_sent.Sentiment) 


from sklearn.metrics import accuracy_score, precision_score, recall_score


pd.crosstab(all_pred_g, review_data_sent.Sentiment)
print("Accuracy: ", accuracy_score(all_pred_g, review_data_sent.Sentiment))
print("Precision : ", precision_score (all_pred_g, review_data_sent.Sentiment, average = 'weighted'))
print("Recall : ", recall_score (all_pred_g, review_data_sent.Sentiment, average = 'weighted'))
print("F1_score : ", f1_score (all_pred_g, review_data_sent.Sentiment, average = 'weighted'))

##### Model building using SVM ##########
from sklearn import  svm

SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(train_tfidf,data_train.Sentiment)
# predict the labels on validation dataset
predictions_SVM = SVM.predict(test_tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, data_test.Sentiment)*100)

predictions_all=SVM.predict(all_tfidf)
print("SVM Accuracy Score -> ",accuracy_score(predictions_all, review_data_sent.Sentiment)*100)

pd.crosstab(predictions_all, review_data_sent.Sentiment)
print("Accuracy: ", accuracy_score(predictions_all, review_data_sent.Sentiment))
print("Precision : ", precision_score (predictions_all, review_data_sent.Sentiment, average = 'weighted'))
print("Recall : ", recall_score (predictions_all, review_data_sent.Sentiment, average = 'weighted'))
print("F1_score : ", f1_score (predictions_all, review_data_sent.Sentiment, average = 'weighted'))

####### model building using XGBM  ######### 

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(train_tfidf, data_train.Sentiment)


# make predictions for test data
y_pred = model.predict(test_tfidf)
print("SVM Accuracy Score -> ",accuracy_score(y_pred, data_test.Sentiment)*100)

predictions_all=model.predict(all_tfidf)
print("SVM Accuracy Score -> ",accuracy_score(predictions_all, review_data_sent.Sentiment)*100)

pd.crosstab(predictions_all, review_data_sent.Sentiment)
print("Accuracy: ", accuracy_score(predictions_all, review_data_sent.Sentiment))
print("Precision : ", precision_score (predictions_all, review_data_sent.Sentiment, average = 'weighted'))
print("Recall : ", recall_score (predictions_all, review_data_sent.Sentiment, average = 'weighted'))
print("F1_score : ", f1_score (predictions_all, review_data_sent.Sentiment, average = 'weighted'))











