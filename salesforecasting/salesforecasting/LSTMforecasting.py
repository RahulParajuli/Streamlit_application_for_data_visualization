#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizers import Adam
from keras import utils
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#%%
#write a function to forecast my sales using LSTM
def sales_using_LSTM(df):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import train_test_split

    df = df[["Created at", "Paid Price"]]
    df = df.rename(columns={"Created at": "Date", "Paid Price": "Sales"})
    #convert the date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    #set the date column as the index
    df = df.set_index('Date')
    #group the data by date and sum the sales
    df = df.groupby(pd.Grouper(freq='D')).sum()
    #plot the data
    df.plot()
    plt.show()
    #split the data into train and test
    train, test = train_test_split(df, test_size=0.2)
    #scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    train = scaler.fit_transform(train)
    test = scaler.transform(test)
    return train, test, scaler, df
#%%
#convert the data into a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df
#%%
def LSTM(train, test):
#convert the data into a supervised learning problem
    train = timeseries_to_supervised(train, 1)
    test = timeseries_to_supervised(test, 1)
    #split the data into X and y
    X_train, y_train = train.iloc[:, :-1], train.iloc[:, -1]
    X_test, y_test = test.iloc[:, :-1], test.iloc[:, -1]
    #reshape the data
    X_train = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])
    return X_train, y_train, X_test, y_test
#%%
def model(X_train, y_train, X_test, y_test):
    #build the model
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    #fit the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=72, validation_data=(X_test, y_test), verbose=2, shuffle=False)
    #plot the loss
    plt.plot(history.history['loss'], label='train')
    # print(history.history['loss'])
    # print(history.history['val_loss'])
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
    #make the prediction
    yhat = model.predict(X_test)
    print(yhat)
    #invert the scaling
    X_test = X_test.reshape(np.array(X_test.shape[0]), np.array(X_test.shape[2]))
    inv_yhat = np.concatenate((np.array(yhat), np.array(X_test[:, 1:])), axis=1)
    print(inv_yhat)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    #invert the scaling for y_test
    print(len(y_test))
    y_test = y_test.shape(len(y_test), 1)
    inv_y = np.concatenate((np.array(y_test),np.array(X_test[:, 1:])), axis=1)
    inv_y = scaler.inverse_transform(np.array(inv_y))
    inv_y = inv_y[:, 0]
    #calculate the RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    #plot the actual and predicted values
    plt.plot(inv_y, label='actual')
    plt.plot(inv_yhat, label='predicted')
    plt.legend()
    plt.show()
    return {{"Test RMSE": rmse}, {"Actual": inv_y}, {"Predicted": inv_yhat}}
#%%
if __name__ == "__main__":
    df = pd.read_csv('/home/drox/Documents/adeyelta/Adeyelta-model/data/Guitar_shop.csv', sep=";")
    df = sales_using_LSTM(df)
    timeseries_to_supervised(df)
    LSTM(train, test)
    model(X_train, y_train, X_test, y_test)

# %%
