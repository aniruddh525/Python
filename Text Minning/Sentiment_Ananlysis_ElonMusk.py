# -*- coding: utf-8 -*-
"""
Created on Thu May 27 19:26:04 2021

@author: anirudh.kumar.verma
"""

import numpy as np 
import pandas as pd 
import string 
import spacy 
from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import re 
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud

## Load data
EMusk=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Text Mining\\new\\Elon_musk.csv",error_bad_lines=False)
EMusk.columns
EMusk.info()
EMusk.head()
EMusk.describe()
EMusk.shape

####### data cleaning/preprocessing########

# remove leading and the trailing characters
EMusk_wspaces = [Text.strip() for Text in EMusk.Text] 
# removes empty strings
EMusk_noempty = [Text for Text in EMusk_wspaces if Text] 
EMusk_noempty[0:10]

# Joining the list into one string/text
EMusk_text = ' '.join(EMusk_noempty)


# Removing unwanted symbols incase if exists
EMusk_text1 = re.sub("[^A-Za-z" "]+"," ",EMusk_text).lower()
EMusk_text11=EMusk_text1.lower()
EMusk_text2 = re.sub("[0-9" "]+"," ",EMusk_text11)

# words that contained in Elon musk tweets
EMusk_words = EMusk_text2.split(" ")


#stop_words = stopwords.words('english')


with open("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Text Mining\\new\\stop.txt","r") as sw:
    stopwords = sw.read()

stopwords = stopwords.split("\n")
stopwords.extend([""])

EMusk_words_nostop = [w for w in EMusk_words if not w in stopwords]

# Joinining all the reviews into single paragraph 
EMusk_words_text = " ".join(EMusk_words_nostop)

# WordCloud can be performed on the string inputs. That is the reason we have combined 
# entire reviews into single paragraph
# Simple word cloud


wordcloud_ip = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(EMusk_words_text)

plt.imshow(wordcloud_ip)

# positive words # Choose the path for +ve words stored in system
with open("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Text Mining\\new\\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")
  
poswords = poswords[36:]

# negative words  Choose path for -ve words stored in system
with open("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Text Mining\\new\\negative-words.txt","r") as neg:
  negwords = neg.read().split("\n")

negwords = negwords[37:]

# negative word cloud
# Choosing the only words which are present in negwords
EMusk_words_text_neg = " ".join ([w for w in EMusk_words_nostop if w in negwords])

wordcloud_neg_in_neg = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(EMusk_words_text_neg)

plt.imshow(wordcloud_neg_in_neg)

# Positive word cloud
# Choosing the only words which are present in positive words
EMusk_words_text_pos = " ".join ([w for w in EMusk_words_nostop if w in poswords])
wordcloud_pos_in_pos = WordCloud(
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(EMusk_words_text_pos)

plt.imshow(wordcloud_pos_in_pos)



