# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:22:16 2021

@author: anirudh.kumar.verma
"""

from flask import Flask,render_template,url_for,request
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Project P58\\Final Code\\review_data.csv", encoding="latin-1")
    df['label'] = df['Sentiment'].map({'Negative': 1, 'Positive': 0})
    df.info()
    df.dropna(inplace=True)
    x = df['Text']
    y = df['label']

    tvec = TfidfVectorizer()
    X=tvec.fit_transform(x.values.astype('U'))
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
 	
    model = LogisticRegression (solver = "lbfgs")

    model.fit(X_train, y_train)

    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = tvec.transform(data).toarray()
        my_prediction = model.predict(vect)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)
