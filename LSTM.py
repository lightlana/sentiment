#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 15:46:49 2018

@author: lightlina
"""
# loading libraries

import pandas as pd # data processing
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical as to_categorical
from sklearn.metrics import confusion_matrix 


#loading data with labels for emotios
data = pd.read_csv('clasifier_final.csv', sep =';', encoding='latin1')
data.tidy_content= data.tidy_content.astype(str)
tweets = data['tidy_content']
labels = data['labels']

uni = tweets.unique()

#tokenize tweets
max_fatures = 2000
tokenizer = Tokenizer(num_words=max_fatures, split=' ')
tokenizer.fit_on_texts(tweets.values)
X = tokenizer.texts_to_sequences(tweets.values)
X = pad_sequences(X)

#LSTM model built with Keras
embed_dim = 128
#Nᵢ is the number of input neurons, Nₒ the number of output neurons
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures, embed_dim,input_length=25))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.3))
model.add(Dense(6,activation='softmax'))
model.compile(loss = 'categorical_crossentropy',
              optimizer='adagrad',
              metrics = ['accuracy'])
print(model.summary())

#training model on training dataset
#define batch size, number of epochs

Y = to_categorical(labels, num_classes=None, dtype='float32')
xtrain, xvalid, ytrain, yvalid = train_test_split(X, Y, random_state=42, test_size=0.3)

batch_size = 200
num_epochs = 30
model.fit(xtrain, ytrain, validation_data=(xvalid, yvalid), batch_size=batch_size, epochs=num_epochs)

#validate model and print statistics
loss_and_metrics = model.evaluate(xvalid, yvalid, batch_size=128)
score,acc = model.evaluate(xvalid, yvalid, verbose = 2, batch_size = batch_size)
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))



#export the trained LSTM model
import pickle
save_classifier = open("LSTM_model.sav","wb")
pickle.dump(model, save_classifier)
save_classifier.close()

