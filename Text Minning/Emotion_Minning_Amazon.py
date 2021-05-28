# -*- coding: utf-8 -*-
"""
Created on Thu May 27 21:50:15 2021

@author: anirudh.kumar.verma
"""

import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string # special operations on strings
import spacy # language models
from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud

nokia105_reviews=[]

### Extracting nokia105 reviews from Amazon website ################
for i in range(1,11):
  ip=[]  
  url="https://www.amazon.in/Nokia-105-2019-Single-Black/product-reviews/B07YYNX5X6/ref=cm_cr_getr_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&filterByStar=critical&pageNumber="+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.findAll("span",attrs={"class","a-size-base review-text review-text-content"})# Extracting the content under specific tags  
  for j in range(len(reviews)):
    ip.append(reviews[j].text)  
  nokia105_reviews=nokia105_reviews+ip  # adding the reviews of one page to empty list which in future contains all the reviews


# to strip white spaces
nokia105_reviews=list(map(lambda s: s.strip(), nokia105_reviews))



# # writng reviews in a text file 
# with open("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Text Mining\\new\\nokia105.txt","w",encoding='utf8') as output:
#     output.write(str(nokia105_reviews1))

from nltk import tokenize
sentences = tokenize.sent_tokenize(" ".join(nokia105_reviews))
sentences[5:15]

#load affin lexicon
afinn = pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\My code\\Text Minning\\Afinn.csv", sep=',', encoding='latin-1')
afinn.shape


sent_df = pd.DataFrame(sentences, columns=['sentence'])
sent_df


affinity_scores = afinn.set_index('word')['value'].to_dict()

#Custom function :score each word in a sentence in lemmatised form, 
#but calculate the score for the whole original sentence.
nlp = spacy.load("en_core_web_sm")
sentiment_lexicon = affinity_scores

def calculate_sentiment(text: str = None):
    sent_score = 0
    if text:
        sentence = nlp(text)
        for word in sentence:
            sent_score += sentiment_lexicon.get(word.lemma_, 0)
    return sent_score


sent_df['sentiment_value'] = sent_df['sentence'].apply(calculate_sentiment)


# how many words are in the sentence?
sent_df['word_count'] = sent_df['sentence'].str.split().apply(len)
sent_df['word_count'].head(10)

sent_df.sort_values(by='sentiment_value').tail(10)

# Sentiment score of the whole review
sent_df['sentiment_value'].describe()


sent_df['index']=range(0,len(sent_df))

import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(sent_df['sentiment_value'])

plt.figure(figsize=(15, 10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)

sent_df.plot.scatter(x='word_count', y='sentiment_value', figsize=(8,8), title='Sentence sentiment value to sentence word count')

