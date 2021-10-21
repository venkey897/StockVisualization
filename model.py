import pandas_datareader.data as web
from datetime import datetime as dt
from keras.layers import  LSTM,Dense,Dropout,Activation
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import numpy as np
import pandas as pd
import math
from keras.models import Sequential
import tensorflow as tf
input_data="MSFT"
start1=dt(2018,1,12)
end1=dt(2020,1,12)
data=web.DataReader(input_data,'yahoo',start1,end1)
df=data
data=df.filter(['Close'])
dataset=data.values
trainingdata=math.ceil(len(dataset)*.8)
scaler=MinMaxScaler(feature_range=(0,1))
scaleddata=scaler.fit_transform(dataset)
train_data=scaleddata[0:trainingdata,:]
x_train=[]
y_train=[]
for i in range(60,len(train_data)):
    x_train.append(train_data[i-60:i,0])
    y_train.append(train_data[i,0])
x_train,y_train=np.array(x_train),np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
print(x_train.shape)
model=Sequential()
model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50,return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))
model.compile(optimizer='adam',loss='mean_squared_error')
model.fit(x_train,y_train,batch_size=1,epochs=1)
model.save("hello.model")


test_data=scaleddata[trainingdata-60: , :]
x_test=[]
y_test=[]
y_test=dataset[trainingdata:,:]
for i in range(60,len(test_data)):
    x_test.append(test_data[i-60:i,0])
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))
predictions=model.predict(x_test)
predictions=scaler.inverse_transform(predictions)
print(predictions)
rmse=np.sqrt(np.mean(predictions-y_test)**2)
print(rmse)