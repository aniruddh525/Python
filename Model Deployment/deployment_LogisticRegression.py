# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 20:54:54 2021

@author: anirudh.kumar.verma
"""

# import streamlit as st
# st.title("First ML application")
# st.write("This is my first Machine Learning application")

import pandas as pd
from sklearn.linear_model import LogisticRegression
import streamlit as st

st.title("Model Deployment : Logistic Regression")

st.sidebar.header("User input parameters")

def user_input_features():
    CLMSEX=st.sidebar.selectbox('Gender',('1','0'))
    CLMINSUR=st.sidebar.selectbox('Insurance',('1','0'))
    SEATBELT=st.sidebar.selectbox('SeatBelt',('1','0'))
    CLMAGE=st.sidebar.number_input("Insert the age")
    LOSS=st.sidebar.number_input("Insert Loss")
    data={'CLMSEX':CLMSEX,
          'CLMINSUR':CLMINSUR,
          'SEATBELT':SEATBELT,
          'CLMAGE':CLMAGE,
          'LOSS':LOSS}
    features=pd.DataFrame(data,index=[0])
    return features

df=user_input_features()
st.subheader("User input parameters on main screen")
st.write(df)

claimants=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\claimants.csv")
claimants.drop(["CASENUM"],inplace=True,axis=1)
claimants=claimants.dropna()

X=claimants.iloc[:,[1,2,3,4,5]]
Y=claimants.iloc[:,0]
clf=LogisticRegression()
clf.fit(X,Y)

prediction=clf.predict(df)
prediction_proba=clf.predict_proba(df)

st.subheader('Predicted Result')
st.write('Yes' if prediction == 0 else 'No')

st.subheader('Prediction probability')
st.write(prediction_proba)


















