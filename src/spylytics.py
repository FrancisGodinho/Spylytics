import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import math

"""
S&P500: beta = 1

BETA < -1:
    AHPI: beta = -5.25
    ELTK: beta = -2.76

BETA ~ -1:
    RKDA: beta = -0.82
    DXR: beta = -1.02
    ZM: beta = -1.51

BETA ~ 0:
    AEY: beta = 0.04
    ADSW: beta = 0.27

BETA ~ 1:
    AAPL: beta = 1.23
    ACN: beta = 1.02
    NFLX: beta = 0.97
    ENPH: beta = 1.01

BETA > 1:
    RCL: beta = 2.57
    URI: beta = 2.40
    DXC: beta = 2.41
"""

SCALER = MinMaxScaler(feature_range=(0, 1)) 

def main():
    tickers = ["S&P500", "AHPI", "ELTK", "RKDA", "DXR", "ZM", "AEY", "ADSW", "AAPL", "ACN", "NFLX", "ENPH", "RCL", "URI", "DXC"]
    losses = []
    
    for ticker in tickers:
        losses.append(predict_stock(ticker))
    
    for ticker, loss in zip(tickers, losses):
        print(f'{ticker}, Loss: {"{:.2e}".format(loss)}')

def predict_stock(ticker):
    model = keras.models.load_model("./models/spylytics.h5")
    
    df, x, y = load_data(file=f"../data/{ticker}.csv", num_days=1095)

    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    loss = model.evaluate(x, y)

    predictions = model.predict(x)
    predictions = SCALER.inverse_transform(predictions)
    
    #plot_results(df[60:], predictions)
    return loss

def load_data(file, num_days):

    df = pd.read_csv(file)
    df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1, inplace=True)

    price_history_len = 60

    if len(df) > num_days + price_history_len:
        df = df[len(df) - num_days - price_history_len:]
    
    close_prices = df.filter(['Close']).values
    close_prices = SCALER.fit_transform(close_prices)

    x, y = [], []
    for i in range(price_history_len, len(close_prices)):
        x.append(close_prices[i - price_history_len : i, 0])
        y.append(close_prices[i, 0])

    return df, np.array(x), np.array(y)
    
def plot_results(df, predictions):

    plt.figure(figsize=(16,8))
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD', fontsize=18)

    plt.plot(df["Date"], df["Close"])
    plt.plot(df["Date"], predictions)

    plt.xticks(np.arange(0, len(df['Date']) - 1, len(df['Date']) / 6))
    plt.legend(['Actual Value', 'Prediction'], loc='lower right')

    plt.show()

if __name__ == "__main__":
    main()