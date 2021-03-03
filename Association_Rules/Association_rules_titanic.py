# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:58:26 2021

@author: anirudh.kumar.verma
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
from mlxtend.preprocessing import TransactionEncoder
# conda install -c conda-forge mlxtend

# load data
titanic=pd.read_csv('C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\Titanic.csv')
titanic.columns
titanic.head()
titanic.Survived.value_counts()

# Pre processsing

dummy_titanic=pd.get_dummies(titanic)
dummy_titanic.head()
dummy_titanic.drop(['Unnamed: 0'], axis=1, inplace=True)

# Apriori Algorithm
frequent_itemsets = apriori(dummy_titanic, min_support=0.1, use_colnames=True)
frequent_itemsets

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
rules

# An leverage value of 0 indicates independence. Range will be [-1 1]
# A high conviction value means that the consequent is highly depending on the antecedent and range [0 inf]


rules.sort_values('lift',ascending = False)[0:20]
rules_gt1=rules[rules.lift>1]




