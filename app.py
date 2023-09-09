import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import yfinance as yfin
import datetime as dt
import streamlit as st

yfin.pdr_override()

current_date = dt.datetime.today()
start_date = current_date - pd.DateOffset(years=6)
end_date = current_date


st.title('Stock Trend Predictions')

user_input = st.text_input('Enter Stock Ticker','TSLA')
df = pdr.get_data_yahoo(user_input,start=start_date,end=end_date)

## Describing Data
st.subheader('Data from 2017 to 2023')
st.write(df.describe())

## Visualizations
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)


st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r',label = '100MA')
plt.plot(df.Close,label = 'ACTUAL')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart 100MA & 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100,'r',label = '100MA')
plt.plot(ma200,'g',label = '200MA')
plt.plot(df.Close,label = 'ACTUAL')
plt.legend()
st.pyplot(fig)


## Splitting Data into Training and Testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

## Load my Model

from tensorflow.keras.models import load_model
model = load_model('keras_model.h52')

## Testing Part

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test , y_test = np.array(x_test) , np.array(y_test)

## Making Predictions

y_predicted = model.predict(x_test)

scaler=scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

## Final Graph

st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='Original_price')
plt.plot(y_predicted,'r',label = 'Predicted_price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


st.title('Price Forecast for the Next 30 Days')

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_input = scaler.fit_transform(np.array(data_testing).reshape(-1,1))
x_input = data_input[401:].reshape(1,-1)

temp_input = list(x_input)
temp_input = temp_input[0].tolist()

## Demonstrate prediction for next 30 days

from numpy import array
# Initialize an empty list to store the predicted outputs
lst_output = []
# Set the number of time steps (n_steps) for the LSTM model
n_steps = 100
# Initialize a loop counter
i = 0
# Loop to generate predictions for the next 30 days
while(i<30):
    # If the input sequence has more than 100 points
    if(len(temp_input)>100):
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1,n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = i+1
    else:
        # Pad the input with zeros to make it compatible with reshaping
        while len(temp_input) < n_steps:
            temp_input.insert(0, 0.0)  # Insert zeros at the beginning
        x_input = np.array(temp_input).reshape(1, n_steps, 1)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i, yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = i+1
        
day_new = np.arange(1,101)
day_new = np.arange(101,131)

df1 = scaler.fit_transform(np.array(df['Close']).reshape(-1,1))

st.subheader('Combined Original and 30 days Forecasting data')

# Convert the scaled 'Close' prices (df1) to a list and extend it with the predicted values (lst_output)
fig = plt.figure(figsize = (12,6))
df3 = df1.tolist()
df3.extend(lst_output)
# Plot the combined data starting from index 1500
plt.plot(df3[1400:], color='blue', label='Original Data')
# Add labels to the x and y axes
plt.xlabel('Days')
plt.ylabel('Price')
# Display a legend to differentiate between the original and extended data
plt.legend()
# Show the plotted graph
st.pyplot(fig)

# Inverse-transform the combined data (df3) to revert the scaled values back to their original scale
df3 = scaler.inverse_transform(df3).tolist()

st.subheader('Original and Forecasted Close Prices with 30 Days Forecasting')

# Plot the combined data (original 'Close' prices and 30 days of forecasted prices)
fig = plt.figure(figsize=(12,6))
plt.plot(df3)

# Add a vertical line to indicate the point where original data ends and forecasting begins
plt.axvline(x=len(df['Close']) - 1, color='red', linestyle='--', label='End of Original Data')
# Add labels to the x and y axes
plt.xlabel('Days')
plt.ylabel('Price')
# Display a legend to differentiate between original and forecasted data
plt.legend()
# Show the plotted graph
st.pyplot(fig)



















