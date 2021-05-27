# -*- coding: utf-8 -*-
"""
Created on Mon May 24 18:39:24 2021

@author: anirudh.kumar.verma
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string # special operations on strings
import spacy # language models

from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud

#!python -m spacy download en_core_web_md

# load data
apple=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\apple.txt",error_bad_lines=False)
apple.columns
apple.info()
apple.head()
apple.describe()
apple.shape


####### data cleaning/preprocessing########

# remove both the leading and the trailing characters
apple = [x.strip() for x in apple.x] 
# removes empty strings, because they are considered in Python as False
apple = [x for x in apple if x] 
apple[0:10]

# Joining the list into one string/text
text = ' '.join(apple)
text

#Punctuation
no_punc_text = text.translate(str.maketrans('', '', string.punctuation)) 
#with arguments (x, y, z) where 'x' and 'y'
# must be equal-length strings and characters in 'x'
# are replaced by characters in 'y'. 'z'
# is a string (string.punctuation here)
no_punc_text


#Tokenization
from nltk.tokenize import word_tokenize
import nltk
#nltk.download('punkt')
text_tokens = word_tokenize(no_punc_text)
print(text_tokens[0:50])


len(text_tokens)

#Remove stopwords
import nltk
from nltk.corpus import stopwords
#nltk.download('punkt')
#nltk.download('stopwords')

my_stop_words = stopwords.words('english')
my_stop_words.extend(['the','apple','laptop','mac','ios'])
no_stop_tokens = [word for word in text_tokens if not word in my_stop_words]
print(no_stop_tokens[0:40])


#Noramalize the data
lower_words = [x.lower() for x in no_stop_tokens]
print(lower_words[0:25])


#Stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in lower_words]
print(stemmed_tokens[0:40])



# NLP english language model of spacy library
nlp = spacy.load("en_core_web_sm") 

# lemmas being one of them, but mostly POS, which will follow later
doc = nlp(' '.join(lower_words))
print(doc[0:40])
len(doc)



lemmas = [token.lemma_ for token in doc]
print(lemmas[0:25])


a1=set(lemmas)-set(no_stop_tokens)
a2=set(no_stop_tokens)-set(lemmas)



# Feature Extraction

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(lemmas)
X

print(vectorizer.vocabulary_)

print(vectorizer.get_feature_names()[0:10])
print(X.toarray()[0:10])
a=X.toarray()

print(X.toarray().shape)

# Let's see how can bigrams and trigrams can be included here

vectorizer_ngram_range = CountVectorizer(analyzer='word',ngram_range=(1,3),max_features = 100)
bow_matrix_ngram =vectorizer_ngram_range.fit_transform(lemmas)


print(vectorizer_ngram_range.get_feature_names())
print(bow_matrix_ngram.toarray())
print(bow_matrix_ngram.toarray().shape)

# TFidf vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer_n_gram_max_features = TfidfVectorizer(norm="l2",analyzer='word', ngram_range=(1,3), max_features = 500)
tf_idf_matrix_n_gram_max_features =vectorizer_n_gram_max_features.fit_transform(lemmas)
print(vectorizer_n_gram_max_features.get_feature_names())
print(tf_idf_matrix_n_gram_max_features.toarray())

# Generate wordcloud


# Import packages
import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");
    
    
    
# Generate wordcloud
stopwords = STOPWORDS
stopwords.add('will')
wordcloud = WordCloud(width = 3000, height = 2000, background_color='black', max_words=100,colormap='Set2',stopwords=stopwords).generate(text)
# Plot
plot_cloud(wordcloud)


# Save image
#wordcloud.to_file("wordcloud.png")














































