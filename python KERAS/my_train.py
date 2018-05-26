# -*- coding: utf-8 -*-
"""
Created on Sat May 19 16:13:18 2018

@author: Dawid
"""
#inspired by
# Train sleep predictor from previously built dataset
# Copyright (c) Andreas Urbanski, 2017

import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras import optimizers
import matplotlib.pyplot as plt
import numpy

# Read input data
data = pickle.load(open('sleepdata.pkl', 'rb'))

## Randomly shuffle data
#np.random.shuffle(data)

# Reshape [[a, b, c, d], ...] -> [[[a, b], ...], [[c, d], ...]]
reshape = lambda a: np.split(a, 2, axis=1)

# Split dataset, 85% training data, 15% test data
training_data, test_data = np.split(data, [int(.90 * data.shape[0])])

# Reshape data
training_data = reshape(training_data)
test_data = reshape(test_data)

X_train = training_data[0]
y_train = training_data[1]

X_test = test_data[0]
y_test = test_data[1]

model = Sequential()
model.add(Dense(8, input_dim=2,activation = "relu"))
#model.add(Dropout(0.1))
model.add(Dense(32,activation = "relu"))
#model.add(Dropout(0.1))
model.add(Dense(64,activation = "relu"))
#model.add(Dropout(0.1))
model.add(Dense(2,activation = "sigmoid"))
#model.add(Dropout(0.1))
adam = optimizers.adam(lr = 0.0001)
model.compile(loss='mse', optimizer= adam,metrics=['accuracy'])
model.fit(X_train, y_train, epochs=2000, batch_size=5)

#model = Sequential()
#model.add(Dense(2,input_dim=2))
#model.add(Dense(12))
#model.add(Dense(2))
## Compile model
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
## Fit the model
#model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=5);
## calculate predictions
#predictions = model.predict(X_test)
### round predictions
##rounded = [round(x[0]) for x in predictions]
##print(rounded)

#model = Sequential() 
#model.add(Dense(4, input_dim=2, init='uniform', activation='sigmoid')) 
#model.add(Dense(3, init='uniform', activation='sigmoid')) 
#model.add(Dense(2, init='uniform', activation='linear'))
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#
#model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])
#model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=2) 
#
#predictions = model.predict(X_test) 
#print(predictions)



