# -*- coding: utf-8 -*-
"""
Created on Sun May 27 10:11:34 2018

@author: Dawid
"""

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
#TIME_STEPS = 14
real_sleep_quality = dataset_test.iloc[:, 3:4].values
#
## Getting the predicted stock price of 2017
#dataset_total = dataset_total.iloc[:,3:4].values
#inputs = dataset_total[len(dataset_total) - len(dataset_test) - TIME_STEPS:]
#inputs = inputs.reshape(-1,1)
#inputs = sc.transform(inputs)
#X_test = []
#for i in range(TIME_STEPS, len(inputs)):
#    X_test.append(inputs[i-TIME_STEPS:i, 0])
#X_test = np.array(X_test)
#X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


predicted_sleep_quality = regressor.predict(X_test)
predicted_sleep_quality = sc_quality.inverse_transform(predicted_sleep_quality)

print(history.history.keys())

figure_sign_core = '-TS'+str(TIME_STEPS) + '-batch'+str(BATCH_SIZE)+'-e'+str(EPOCHS)+'-LSTM'+str(LAYER1)+str(LAYER2)+str(LAYER3)+str(LAYER4)
fig2 = plt.figure(2)
fig2.canvas.set_window_title('loss-chart'+figure_sign_core);
plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Visualising the results
fig3 = plt.figure(3)
fig3.canvas.set_window_title('Result-chart' + figure_sign_core);
plt.plot(real_sleep_quality, color = 'red', label = 'Real Sleep Quality')
plt.plot(predicted_sleep_quality, color = 'blue', label = 'Predicted Sleep Quality')
plt.title('LSTM Sleep Quality Prediction')
plt.xlabel('Time [day]')
plt.ylabel('Sleep Quality [%]')
plt.legend()
plt.show()

#