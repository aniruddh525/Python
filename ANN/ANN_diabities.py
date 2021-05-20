# -*- coding: utf-8 -*-
"""
Created on Sat May 15 11:54:07 2021

@author: anirudh.kumar.verma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential,load_model


# load data as array
seed=7
np.random.seed(seed)
diabities=np.loadtxt("C:\\Users\\anirudh.kumar.verma.DIR\\Documents\\personal\\Learning\\DS\\Python\\Data set\\pima-indians-diabetes.data.csv",delimiter=",")
X = diabities[:,0:8]
Y = diabities[:,8]


ann_model = Sequential()
ann_model.add(layers.Dense(50, input_dim=8,activation='relu'))
ann_model.add(layers.Dense(8, activation='relu'))
ann_model.add(layers.Dense(1, activation='sigmoid'))

# Compile model
ann_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Fit the model
ann_model.fit(X, Y, validation_split=0.33, nb_epoch=250, batch_size=10)


# evaluate the model
scores = ann_model.evaluate(X, Y)
print("%s: %.2f%%" % (ann_model.metrics_names[1], scores[1]*100))

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


























