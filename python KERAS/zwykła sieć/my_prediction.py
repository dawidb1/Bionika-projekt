# -*- coding: utf-8 -*-
"""
Created on Sat May 19 18:09:41 2018

@author: Dawid
"""
import numpy
import matplotlib.pyplot as plt

# generate predictions for training
trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)
y_pred = testPredict

# shift train predictions for plotting
trainPredictPlot = numpy.empty([len(data),2])
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[0:len(trainPredict), :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty([len(data),2])
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict):len(data), :] = testPredict

# plot baseline and predictions
temp = numpy.concatenate((y_train, y_test), axis=0)
[time,qual] = plt.plot(temp)
plt.show()

#plt.plot(trainPredictPlot)
[time2,qual2] = plt.plot(testPredictPlot)
plt.legend([time,qual,time2,qual2], ["data_time","data_qual","test_time","test_qual"], loc=1)
plt.show()

score = model.evaluate(X_test, y_test)
print(score)