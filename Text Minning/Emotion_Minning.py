# -*- coding: utf-8 -*-
"""
Created on Mon May 24 18:46:34 2021

@author: anirudh.kumar.verma
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string # special operations on strings
import spacy # language models

from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud


#Sentiment analysis
afinn = pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\My code\\Text Minning\\Afinn.csv", sep=',', encoding='latin-1')
afinn.shape

afinn.head()



apple=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\apple.txt",error_bad_lines=False)


# remove both the leading and the trailing characters
apple = [x.strip() for x in apple.x] 
# removes empty strings, because they are considered in Python as False
apple = [x for x in apple if x] 
apple[0:10]



from nltk import tokenize
sentences = tokenize.sent_tokenize(" ".join(apple))
sentences[5:15]

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

# test that it works
calculate_sentiment(text = 'beuatiful and amazing is bad')


sent_df['sentiment_value'] = sent_df['sentence'].apply(calculate_sentiment)


# how many words are in the sentence?
sent_df['word_count'] = sent_df['sentence'].str.split().apply(len)
sent_df['word_count'].head(10)




sent_df.sort_values(by='sentiment_value').tail(10)

# Sentiment score of the whole review
sent_df['sentiment_value'].describe()




# Sentiment score of the whole review
sent_df[sent_df['sentiment_value']<=0].head()


sent_df[sent_df['sentiment_value']>=20].head()

sent_df['index']=range(0,len(sent_df))



import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(sent_df['sentiment_value'])



plt.figure(figsize=(15, 10))
sns.lineplot(y='sentiment_value',x='index',data=sent_df)


sent_df.plot.scatter(x='word_count', y='sentiment_value', figsize=(8,8), title='Sentence sentiment value to sentence word count')




