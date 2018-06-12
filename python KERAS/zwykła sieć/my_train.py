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


# FIRST ANN
model = Sequential()
model.add(Dense(8, input_dim=2,activation = "relu"))
model.add(Dropout(0.1))
model.add(Dense(32,activation = "relu"))
model.add(Dropout(0.1))
model.add(Dense(64,activation = "relu"))
model.add(Dropout(0.1))
model.add(Dense(2,activation = "sigmoid"))
model.add(Dropout(0.1))
adam = optimizers.adam(lr = 0.0001)
model.compile(loss='mse', optimizer= adam,metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=5)




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

# DIFFERNET COMPILERS
#model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy'])


# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    model = Sequential()
    model.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 2))
    model.add(Dense(units = 32, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))
    model.add(Dense(units = 2, kernel_initializer = 'uniform', activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])
    return model

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train)
mean = accuracies.mean()
variance = accuracies.std()

#K_FOLD MANUAL
# define 10-fold cross validation test harness
from sklearn.model_selection import StratifiedKFold
import numpy

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
X = numpy.concatenate((X_train, X_test), axis=0)
Y = numpy.concatenate((y_train, y_test), axis=0)
for train, test in kfold.split(X, Y):
  # create model
	model = Sequential()
	model.add(Dense(12, input_dim=8, activation='relu'))
	model.add(Dense(8, activation='relu'))
	model.add(Dense(2, activation='sigmoid'))
	# Compile model
	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
	# Fit the model
	model.fit(X_train, y_train, epochs=150, batch_size=10, verbose=0)
	# evaluate the model
	scores = model.evaluate(X_test, y_test, verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))
