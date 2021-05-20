# -*- coding: utf-8 -*-
"""
Created on Thu May 20 13:06:54 2021

@author: anirudh.kumar.verma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential,load_model
from sklearn import preprocessing


# load data as array


forestfires=pd.read_csv("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Assisgnments\\ANN\\forestfires.csv")
forestfires=forestfires[['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind',
       'rain', 'area', 'dayfri', 'daymon', 'daysat', 'daysun', 'daythu',
       'daytue', 'daywed', 'monthapr', 'monthaug', 'monthdec', 'monthfeb',
       'monthjan', 'monthjul', 'monthjun', 'monthmar', 'monthmay', 'monthnov',
       'monthoct', 'monthsep', 'size_category']]
forestfires.columns
forestfires.head()
forestfires.info()
forestfires.describe()

# put label for categorical output variable

label_encoder = preprocessing.LabelEncoder()
forestfires['size_category']= label_encoder.fit_transform(forestfires['size_category']) 

# convert data frame into array
forestfires_arr=forestfires.values
X = forestfires_arr[:,0:28]
Y = forestfires_arr[:,28]


ann_model = Sequential()
ann_model.add(layers.Dense(28, input_dim=28,activation='relu'))
ann_model.add(layers.Dense(8, activation='relu'))
ann_model.add(layers.Dense(1, activation='sigmoid'))

# Compile model
ann_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
ann_model.fit(X, Y, validation_split=0.33, nb_epoch=100, batch_size=10)


# evaluate the model
scores = ann_model.evaluate(X, Y)
print("%s: %.2f%%" % (ann_model.metrics_names[1], scores[1]*100))
#acc: 94.58%

# Visualize training history

# list all data in history
ann_model.history.history.keys()


# summarize history for accuracy
plt.plot(ann_model.history.history['acc'])
plt.plot(ann_model.history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(ann_model.history.history['loss'])
plt.plot(ann_model.history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

