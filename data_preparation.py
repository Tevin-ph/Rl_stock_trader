import yfinance as yf
import pandas as pd
import numpy as np
import talib  # TA-Lib for indicators
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Download AAPL data
ticker = 'AAPL'
train_data = yf.download(ticker, start='2010-01-01', end='2020-12-31')
test_data = yf.download(ticker, start='2021-01-01', end='2023-12-31')

# Keep OHLCV
train_data = train_data[['Open', 'High', 'Low', 'Close', 'Volume']]
test_data = test_data[['Open', 'High', 'Low', 'Close', 'Volume']]

# Add technical indicators with TA-Lib
def add_features(df):
 # Ensure we pass a 1-D float64 numpy array to TA-Lib (some pandas slices can be shaped (n,1))
 close_arr = np.asarray(df['Close'], dtype='float64').ravel()
 df['RSI'] = talib.RSI(close_arr, timeperiod=14)
 macd, macd_signal, macd_hist = talib.MACD(close_arr, fastperiod=12, slowperiod=26, signalperiod=9)
 df['MACD'] = macd
 df['MACD_signal'] = macd_signal
 df['MACD_hist'] = macd_hist
 bb_upper, bb_middle, bb_lower = talib.BBANDS(close_arr, timeperiod=20, nbdevup=2, nbdevdn=2)
 df['BB_upper'] = bb_upper
 df['BB_middle'] = bb_middle
 df['BB_lower'] = bb_lower
 df['SMA_20'] = talib.SMA(close_arr, timeperiod=20)
 df['EMA_12'] = talib.EMA(close_arr, timeperiod=12)
 return df

train_data = add_features(train_data)
test_data = add_features(test_data)

# Normalize (fit scaler on train only, apply to both)
feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD', 'MACD_signal', 'BB_upper', 'BB_lower', 'SMA_20', 'EMA_12']
scaler = MinMaxScaler()
train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])
test_data[feature_columns] = scaler.transform(test_data[feature_columns])  # Use train's scaler!

# Drop NaNs
train_data = train_data.dropna()
test_data = test_data.dropna()

# Save
train_data.to_csv('aapl_train.csv', index=False)
test_data.to_csv('aapl_test.csv', index=False)
print(f"Train data: {len(train_data)} rows saved.")
print(f"Test data: {len(test_data)} rows saved.")
print("Features added: OHLCV + RSI, MACD, Bollinger Bands, SMA_20, EMA_12")