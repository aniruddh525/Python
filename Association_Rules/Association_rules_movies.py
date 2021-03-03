# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:00:23 2021

@author: anirudh.kumar.verma
"""

import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules

# load data
movies=pd.read_csv('C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Association Rules\\my_movies.csv')
movies.columns
movies.head()

# selecting required columns
movies1=movies.iloc[:,5:]


# Apriori Algorithm
frequent_itemsets = apriori(movies1, min_support=0.2, use_colnames=True)
frequent_itemsets

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
rules


rules.sort_values('lift',ascending = False)
# all 15 rows have lift ratio greater than 1.

# trying with other support and confidence
frequent_itemsets1 = apriori(movies1, min_support=0.1, use_colnames=True)
frequent_itemsets1

rules1 = association_rules(frequent_itemsets1, metric="confidence", min_threshold=0.5)
rules1

rules1.sort_values('lift',ascending = False)[0:20]
rules_gt1=rules1[rules1.lift>1]

# we get 209 actionable rules (lift > 1) if support is decresed from 0.2 to 0.1

# increasing confidence for support 0.1

rules2 = association_rules(frequent_itemsets1, metric="confidence", min_threshold=0.7)
rules2

rules2.sort_values('lift',ascending = False)[0:20]
rules_gt2=rules2[rules2.lift>1]

# all 129 rows have lift ratio greater than 1. 



