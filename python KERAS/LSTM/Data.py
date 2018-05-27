# -*- coding: utf-8 -*-
"""
Created on Sun May 27 08:55:29 2018

@author: Dawid
"""

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime 

#DEFINE FUNCTIONS
def getStartMinutesArray(dataset_train):
    start_times = pd.DatetimeIndex(dataset_train['Sleep start']) 
    start_minutes = (start_times.hour * 60 + start_times.minute).values
    
    start_minutes = start_minutes.reshape(-1,1)
    #change after midnight values (add a day)
    for time in range(start_minutes.shape[0]):
        if(start_minutes[time] < 16 * 60):
            start_minutes[time] = start_minutes[time] + 24*60
    return start_minutes
def getEndMinutesArray(dataset_train):
    end_times = pd.DatetimeIndex(dataset_train['Sleep end']) 
    end_minutes = (end_times.hour * 60 + end_times.minute).values    
    end_minutes = end_minutes.reshape(-1,1)
    return end_minutes
def getDayOfWeekArray(dataset_train):
    day = []
    for date in dataset_train['Date']:
        full_date = datetime.datetime.strptime(date, '%d.%m.%Y').date()
        day.append(full_date.weekday())
    day = np.array(day)
    day = day.reshape(-1,1)
    return day
# Importing the training set
dataset_total = pd.read_csv('sleep_data_sort_kropka.csv', sep=';')
dataset_train, dataset_test = np.split(dataset_total, [int(.90 * dataset_total.shape[0])])

start_minutes = getStartMinutesArray(dataset_train)
end_minutes = getEndMinutesArray(dataset_train)
day_of_week = getDayOfWeekArray(dataset_train)
quality =  dataset_train.iloc[:, 3:4].values

training_set = np.c_[start_minutes, end_minutes, quality]

training_set = dataset_train.iloc[:, 3:4].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler

#define scalers
sc_quality = MinMaxScaler(feature_range = (0, 1))
sc_start = MinMaxScaler(feature_range = (0, 1))
sc_end = MinMaxScaler(feature_range = (0, 1))
sc_dayOfWeek = MinMaxScaler(feature_range=(0,1))

#scale data
start_scaled = sc_start.fit_transform(start_minutes)
end_scaled = sc_end.fit_transform(end_minutes)
day_of_week = sc_dayOfWeek.fit_transform(day_of_week)
quality_scaled = sc_quality.fit_transform(quality)

#training_set_scaled = sc.fit_transform(training_set)
start_inverse = sc_start.inverse_transform(start_scaled)
end_inverse = sc_end.inverse_transform(end_scaled)
quality_inverse = sc_quality.inverse_transform(quality_scaled)

# Creating a data structure with 14 start_timesteps and 1 output
TIME_STEP = 14;
X_train_start = []
X_train_end = []
X_train_weekday = []
y_train = []

for i in range(TIME_STEP, start_scaled.shape[0]):
    X_train_start.append(start_scaled[i-TIME_STEP:i, 0])
    X_train_end.append(end_scaled[i-TIME_STEP:i, 0])
    X_train_weekday.append(quality_scaled[i-TIME_STEP:i, 0])

    y_train.append(quality_scaled[i, 0])    
    
X_train_start, X_train_end, X_train_weekday, = np.array(X_train_start), np.array(X_train_end), np.array(X_train_weekday)

# Reshaping
X_train = np.dstack((X_train_start, X_train_end, X_train_weekday))


#plot data
real_sleep_quality_data = dataset_total.iloc[:,3:4].values
plt.plot(real_sleep_quality_data, color = 'red', label = 'Real Sleep Quality')
plt.title('Real Sleep Quality')
plt.xlabel('Time [day]')
plt.ylabel('Sleep Quality [%]')
plt.legend()
plt.show()

