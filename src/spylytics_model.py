import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math

SCALER = MinMaxScaler(feature_range=(0, 1)) 

def main():
    """
    Creates an LSTM model, trains it based on S&P500 data, saves the model, 
    and then plots the results of the session.
    """
    price_history_len = 60
    df, x_train, y_train, x_test, y_test = create_datasets(price_history_len)
   
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #create model
    model = create_model((x_train.shape[1], 1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=10)

    model.evaluate(x_test, y_test)
    model.save("./models/spylytics_model.h5")

    #predict
    predictions = model.predict(x_test) 
    predictions = SCALER.inverse_transform(predictions)

    plot_results(df[price_history_len:], predictions)

def create_datasets(price_history_len):
    """
    Creates both x, y (input, output) testing and training datasets from the 
    S&P dataset. The training set is 80% of the entire dataset.

    @param price_history_len: The length of the prices in the input datasets.

    @returns: Testing and training datasets. The input set consists of a list 
    of the previous 'price_history_len' prices of a stock, and the output set 
    consists of the price after the 'price_history_len' of the stock.
    """

    df = pd.read_csv('./../data/S&P500_data.csv')[18080:] # get data from 01/03/2000 onwards 18080
    df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)
    
    close_prices = df.filter(['Close']).values
    close_prices = SCALER.fit_transform(close_prices)

    train_data_len = math.ceil(len(close_prices) * 0.8)
    x, y = [], []

    for i in range(price_history_len, len(close_prices)):
        x.append(close_prices[i - price_history_len : i, 0])
        y.append(close_prices[i, 0])

    x, y = np.array(x), np.array(y)

    return df, x[:train_data_len], y[:train_data_len], x[train_data_len:], y[train_data_len:]

def create_model(input_shape):
    """
    Creates the tensorflow model.

    @param input_shape: The input shape of the first layer based on the price_history_len.

    @returns: A tensorflow model.
    """
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(keras.layers.LSTM(units=50, return_sequences=False))
    model.add(keras.layers.Dense(25))
    model.add(keras.layers.Dense(1))
    return model

def plot_results(df, predictions):
    """
    Plots the results in two graphs. The first graph shows the training data, 
    testing data and actual values. The second graph is a zoomed in version of 
    the first which shows just the training data and actual values.

    @param df: The dataframe which contains all the actual data from the S&P500.
    @param predictions: The resulting predictions of the testing data.
    """
    plt.figure(figsize=(16,8))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD', fontsize=18)

    train_len = len(df["Date"]) - len(predictions)

    plt.plot(df['Date'][:train_len], df['Close'][:train_len])
    plt.plot(df['Date'][train_len:], df['Close'][train_len:])
    plt.plot(df['Date'][train_len:], predictions)
    
    plt.xticks(np.arange(0, len(df['Date']) - 1, len(df['Date'][:train_len]) / 6))
    plt.legend(['Train', 'Actual Value', 'Prediction'], loc='lower right')

    #zoomed in plot
    plt.figure(figsize=(16,8))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD', fontsize=18)

    plt.plot(df['Date'][train_len:], df['Close'][train_len:])
    plt.plot(df['Date'][train_len:], predictions)

    plt.xticks(np.arange(0, len(df['Date'][train_len:]) - 1, len(df['Date'][train_len:]) / 6))
    plt.legend(['Actual Value', 'Prediction'], loc='lower right')

    plt.show()


if __name__ == "__main__":
    main()



