# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset = pd.read_csv('KOTAKBANK.NS.csv')
missing_values = dataset[dataset.isna().any(axis = 1)]
dataset_imputed = dataset.dropna().reset_index(drop = True)
training_set = dataset_imputed.iloc[0:1232, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1232):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train_reshaped = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train_reshaped.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train_reshaped, y_train, epochs = 100, batch_size = 32)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2018
## dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_imputed.iloc[1232:, 1:2].values

# Getting the predicted stock price of 2018
## dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
inputs = dataset_imputed.iloc[len(training_set) - 60:, 1:2].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 153):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
dates = dataset_imputed.iloc[len(training_set)::22, 0:1].values
plt.xticks(range(0, 153, 22), dates, rotation = 45)
plt.plot(real_stock_price, color = 'red', label = 'Real Kotak Mahindra Bank Stock Price (2018)')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Kotak Mahindra Bank Stock Price(2018)')
plt.title('Kotak Mahindra Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Kotak Mahindra Bank Stock Price')
plt.legend()
plt.show()

regressor.save('model_kotakbank.h5')
