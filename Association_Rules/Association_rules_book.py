# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:12:03 2021

@author: anirudh.kumar.verma
"""
import pandas as pd
from mlxtend.frequent_patterns import apriori,association_rules
import matplotlib.pyplot as plt

# load data
book=pd.read_csv('C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\Association Rules\\book.csv')
book.columns
book.head()

# Apriori Algorithm
frequent_itemsets1 = apriori(book, min_support=0.3, max_len=3,use_colnames=True)
frequent_itemsets1

rules1 = association_rules(frequent_itemsets1, metric="confidence", min_threshold=0.5)
rules1.sort_values('lift',ascending = False)

# chcek for other support  values
frequent_itemsets2 = apriori(book, min_support=0.1, max_len=3,use_colnames=True)
frequent_itemsets2

rules2 = association_rules(frequent_itemsets2, metric="confidence", min_threshold=0.5)
rules2.sort_values('lift',ascending = False)


# chcek for other support  values
frequent_itemsets3 = apriori(book, min_support=0.05, max_len=3,use_colnames=True)
frequent_itemsets3

rules3 = association_rules(frequent_itemsets3, metric="confidence", min_threshold=0.5)
rules3.sort_values('lift',ascending = False)

# chcek for other support  values
frequent_itemsets4 = apriori(book, min_support=0.03, max_len=3,use_colnames=True)
frequent_itemsets4

rules4 = association_rules(frequent_itemsets4, metric="confidence", min_threshold=0.5)
rules4.sort_values('lift',ascending = False)

# chcek for other support  values
frequent_itemsets5 = apriori(book, min_support=0.01, max_len=3,use_colnames=True)
frequent_itemsets5

rules5 = association_rules(frequent_itemsets5, metric="confidence", min_threshold=0.5)
rules5.sort_values('lift',ascending = False)

#Plot for support values v no of rules

plt.plot([0.3,0.1,0.05,0.03,0.01],[len(rules1),len(rules2),len(rules3),len(rules4),len(rules5)], 'r--')



# Apriori Algorithm
frequent_itemsets1 = apriori(book, min_support=0.1, max_len=3,use_colnames=True)
frequent_itemsets1

rules1 = association_rules(frequent_itemsets1, metric="confidence", min_threshold=0.5)
rules1.sort_values('lift',ascending = False)

# chcek for other confidence values
frequent_itemsets2 = apriori(book, min_support=0.1, max_len=3,use_colnames=True)
frequent_itemsets2

rules2 = association_rules(frequent_itemsets2, metric="confidence", min_threshold=0.4)
rules2.sort_values('lift',ascending = False)


# chcek for other confidence values
frequent_itemsets3 = apriori(book, min_support=0.1, max_len=3,use_colnames=True)
frequent_itemsets3

rules3 = association_rules(frequent_itemsets3, metric="confidence", min_threshold=0.3)
rules3.sort_values('lift',ascending = False)

# chcek for other confidence values
frequent_itemsets4 = apriori(book, min_support=0.1, max_len=3,use_colnames=True)
frequent_itemsets4

rules4 = association_rules(frequent_itemsets4, metric="confidence", min_threshold=0.2)
rules4.sort_values('lift',ascending = False)

# chcek for other confidence values
frequent_itemsets5 = apriori(book, min_support=0.1, max_len=3,use_colnames=True)
frequent_itemsets5

rules5 = association_rules(frequent_itemsets5, metric="confidence", min_threshold=0.1)
rules5.sort_values('lift',ascending = False)

#Plot for conf. values v no of rules

plt.plot([0.5,0.4,0.3,0.2,0.1],[len(rules1),len(rules2),len(rules3),len(rules4),len(rules5)], 'r--')


