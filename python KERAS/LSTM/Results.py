# -*- coding: utf-8 -*-
"""
Created on Sun May 27 10:11:34 2018

@author: Dawid
"""

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
TIME_STEPS = 14

real_sleep_quality = dataset_test.iloc[:, 3:4].values

# Getting the predicted stock price of 2017
dataset_total = dataset_total.iloc[:,3:4].values
inputs = dataset_total[len(dataset_total) - len(dataset_test) - TIME_STEPS:]
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(TIME_STEPS, len(inputs)):
    X_test.append(inputs[i-TIME_STEPS:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_sleep_quality = regressor.predict(X_test)
predicted_sleep_quality = sc.inverse_transform(predicted_sleep_quality)

# Visualising the results
plt.plot(real_sleep_quality, color = 'red', label = 'Real Sleep Quality')
plt.plot(predicted_sleep_quality, color = 'blue', label = 'Predicted Sleep Quality')
plt.title('LSTM Sleep Quality Prediction')
plt.xlabel('Time [day]')
plt.ylabel('Sleep Quality [%]')
plt.legend()
plt.show()