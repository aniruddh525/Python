# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:22:16 2021

@author: anirudh.kumar.verma
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import string # special operations on strings
import spacy # language models

from matplotlib.pyplot import imread
from matplotlib import pyplot as plt
from wordcloud import WordCloud



# load data
apple=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\apple.txt",error_bad_lines=False)
apple.columns
apple.info()
apple.head()
apple.describe()
apple.shape

# remove both the leading and the trailing characters
apple = [x.strip() for x in apple.x] 
# removes empty strings, because they are considered in Python as False
apple = [x for x in apple if x] 
apple[0:10]

##Part Of Speech Tagging
from IPython.core.display import display, HTML
from pathlib import Path
nlp = spacy.load("en_core_web_sm")

one_block = apple[2]
doc_block = nlp(one_block)
html=spacy.displacy.render(doc_block, style='ent',page=True,minify=True) # in jupyter , can use Jupyter=True

output_path = Path("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\My code\\Text Minning\\Emo_min.svg")
output_path.open("w", encoding="utf-8").write(html)

one_block

for token in doc_block[0:20]:
    print(token, token.pos_)
    
    
#Filtering for nouns and verbs only
nouns_verbs = [token.text for token in doc_block if token.pos_ in ('NOUN', 'VERB')]
print(nouns_verbs[5:25])



#Counting tokens again
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

X = cv.fit_transform(nouns_verbs)
sum_words = X.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in cv.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
wf_df = pd.DataFrame(words_freq)
wf_df.columns = ['word', 'count']

wf_df[0:10]


##Visualizing results
#Barchart for top 10 nouns + verbs
wf_df[0:10].plot.bar(x='word', figsize=(12,8), title='Top verbs and nouns')

# EMotion minning
