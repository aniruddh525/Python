# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 13:42:06 2021

@author: anirudh.kumar.verma
"""

import pandas as pd
from lifelines import KaplanMeierFitter

unemployment=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\survival_unemployment.csv")

unemployment.head()

unemployment.spell.value_counts()
unemployment.event.value_counts()

unemployment["spell"].describe()

# spell is referred as time here 
T=unemployment.spell

# model building
kmf=KaplanMeierFitter()
kmf.fit(T,event_observed=unemployment.event)
kmf.plot()

# for multiple groups where group is ui

# applying KaplanMeierFitter model on time and events for the group 1
kmf.fit(T[unemployment.ui==1],unemployment.event[unemployment.ui==1],label='1')
ax=kmf.plot()

#for group 0
kmf.fit(T[unemployment.ui==0],unemployment.event[unemployment.ui==0],label='0')
kmf.plot(ax=ax)

