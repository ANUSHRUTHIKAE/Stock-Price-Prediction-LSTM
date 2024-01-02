import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

import math
from sklearn.metrics import mean_squared_error


time_step = 100

def convert_dataframe(data, time_step):
    x, y = [], []
    for i in range(len(data)-time_step-1):
        
        #i=0,time_step=3 :x[0]=0,1,2 y[0]=3, x[1]=1,2,3 y[1]=4 .....
        x.append(data[i:(i+time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(x), np.array(y)

def stock_predict(ticker,stock):
    
    # Read data from tiingo
    path = 'data'+ticker+'.csv'
    df = pd.read_csv(path)
    df =df.reset_index()[stock]

    # Standardisation
    scaler=MinMaxScaler(feature_range=(0,1))
    df=scaler.fit_transform(np.array(df).reshape(-1,1))
    
    # Train Test Split
    train_size=int(len(df)*0.65)
    test_size=len(df)-train_size
    train_data,test_data=df[0:train_size,:],df[train_size:len(df),:1]

    # Convert to data frame
    X_train, y_train = convert_dataframe(train_data, time_step)
    X_test, y_test = convert_dataframe(test_data, time_step)
    
    # Reshape the data frame to feed into LSTM [number of samples, time steps,number of features]
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(100,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    
    model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=1,batch_size=64,verbose=1)
    
    # predicting for next 30 days using previous time step days
    l = len(test_data)-100
    x_input=test_data[l:].reshape(1,-1)
    x_input = x_input[0]
    forecast=[]
    i=0
    n_steps = 100
    while i < 30:
        model_input = x_input[i:i+n_steps].reshape((1, n_steps,1))
        y_hat = model.predict(model_input, verbose=0)
        forecast.append(y_hat[0].tolist())
        x_input=np.hstack([x_input,y_hat[0]])
        i+=1
        
    day_new=np.arange(1,101)
    day_pred=np.arange(101,131)
    plt.switch_backend('Agg')
    # # plot for past 100 days 
    plt.plot(day_new,scaler.inverse_transform(test_data[l:]))
    # plot for forecast next 30 days
    plt.plot(day_pred,scaler.inverse_transform(forecast))
    plt.savefig("static/images/plot_100.jpg")
    plt.close()
        
    day_new=np.arange(1,len(df)+1)
    day_pred=np.arange(len(df)+1,len(df)+31)
    
    # plot for past days (all the data in df_close)
    plt.switch_backend('Agg')
    plt.plot(day_new,scaler.inverse_transform(df))
    # plot for forecast next 30 days
    plt.plot(day_pred,scaler.inverse_transform(forecast))
    plt.savefig("static/images/plot_full.jpg")
    plt.close()

# stock_predict('AAPL','close') 