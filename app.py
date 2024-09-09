import pandas as pd
import datetime as dt
from datetime import date
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import tensorflow as tf
import streamlit as st
from keras.models import load_model

start = '2010-01-01'
end = date.today().strftime("%Y-%m-%d")

st.title("Stock Price Prediction")

user_input = st.text_input('Enter Stock Ticker','TCS.NS')
st.subheader('Data from 2010 to till today')
#load the dataset
def load_data(ticker):
    data = yf.download(ticker, start, end)
    data.reset_index(inplace=True)
    return data

data = load_data(user_input)
df = data
st.write(df.describe())

#visuallisation of the data
st.subheader('closing Price vs Time chart')
fig = plt.figure(figsize=(12, 6))
plt.title(user_input)
plt.plot(df['Close'])
plt.grid(True)
st.pyplot(fig)


#ploting for 100 min avg
st.subheader('closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
plt.grid(True)
st.pyplot(fig)


#ploting for 100 min avg
st.subheader('closing Price vs Time chart with 100MA and 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(df.Close)
plt.grid(True)
st.pyplot(fig)


#splitting data into training and testing 
train = pd.DataFrame(data[0:int(len(data)*0.70)])
test = pd.DataFrame(data[int(len(data)*0.70): int(len(data))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

train_close = train.iloc[:, 4:5].values
test_close = test.iloc[:, 4:5].values

data_training_array = scaler.fit_transform(train_close)

#loading the moedl

model = load_model('keras_model.h5')

#testing part

past_100_days = pd.DataFrame(train_close[-100:])
test_df = pd.DataFrame(test_close)
final_df = pd.concat([past_100_days, test_df], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
   x_test.append(input_data[i-100: i])
   y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)   
y_pred = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler
y_pred = y_pred * scale_factor
y_test = y_test * scale_factor

st.subheader('Making prediction and plotting the graph of predicted vs actual values')
fig = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = "Original Price")
plt.plot(y_pred, 'r', label = "Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
st.pyplot(fig)
