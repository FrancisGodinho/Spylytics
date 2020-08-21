import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math


def create_datasets(close_prices, price_history_len):

    train_data_len = math.ceil(len(close_prices) * 0.8)
    x, y = [], []

    for i in range(price_history_len, len(close_prices)):
        x.append(close_prices[i - price_history_len : i, 0])
        y.append(close_prices[i, 0])

    x, y = np.array(x), np.array(y)

    return x[:train_data_len], y[:train_data_len], x[train_data_len:], y[train_data_len:]

def create_model(input_shape):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(keras.layers.LSTM(units=50, return_sequences=False))
    model.add(keras.layers.Dense(25))
    model.add(keras.layers.Dense(1))
    return model

def plot_results(df, predictions):
    plt.figure(figsize=(16,8))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD', fontsize=18)

    train_len = len(df["Date"]) - len(predictions)

    plt.plot(df['Date'][:train_len], df['Close'][:train_len])
    plt.plot(df['Date'][train_len:], df['Close'][train_len:])
    plt.plot(df['Date'][train_len:], predictions)
    
    plt.xticks(np.arange(0, len(df['Date']) - 1, 1000))
    plt.legend(['Train', 'Actual Value', 'Prediction'], loc='lower right')

    #zoomed in plot
    plt.figure(figsize=(16,8))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD', fontsize=18)

    plt.plot(df['Date'][train_len:], df['Close'][train_len:])
    plt.plot(df['Date'][train_len:], predictions)

    plt.xticks(np.arange(0, len(df['Date'][train_len:]) - 1, 200))
    plt.legend(['Actual Value', 'Prediction'], loc='lower right')

    plt.show()

def main():
    df = pd.read_csv('../data/S&P500_data.csv')[18080:] # get data from 01/03/2000 onwards 18080
    df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)
    print("Dataset length:", len(df["Close"]))
    close_price_df = df.filter(['Close'])
    close_prices = close_price_df.values

    scaler = MinMaxScaler(feature_range=(0, 1)) 
    data = scaler.fit_transform(close_prices)

    price_history_len = 60
    x_train, y_train, x_test, y_test = create_datasets(data, price_history_len)
   
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    #create model
    model = create_model((x_train.shape[1], 1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, batch_size=1, epochs=3)

    model.evaluate(x_test, y_test)

    #predict
    predictions = model.predict(x_test) 
    predictions = scaler.inverse_transform(predictions)

    plot_results(df, predictions)
    

main()



