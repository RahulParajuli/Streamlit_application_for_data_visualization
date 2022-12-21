#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from warnings import filterwarnings
filterwarnings('ignore')
from scipy.special import boxcox1p, inv_boxcox
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
import scipy.stats as scs
from itertools import product

#%%
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

#%%
def tsplot(y, lags=None, figsize=(10,8), system= "bmh"):
    #plot time series
    if not isinstance(y,pd.Series):
        y = pd.Series(y)
    with plt.style.context(system):
        fig = plt.figure(figsize=figsize)
        layout = (2,2)
        ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
        hist_ax = plt.subplot2grid(layout, (1,0))
        acf_ax = plt.subplot2grid(layout, (1,1))
        y.plot(ax=ts_ax)
        ts_ax.set_title('Time Series Analysis Plots')
        y.plot(ax=hist_ax, kind='hist', bins=25)
        hist_ax.set_title('Histogram')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax, alpha=0.5)
        plt.tight_layout()
    return

#%%
def display_menu():
    option = input("Enter your operation\n1. forcast my sales\n2. Exit\n ")
    return option

#%%
def sales_forecast(df):
    option = display_menu()
    if option == "1":
        forecast = forecast_sales(df)
        print(forecast)
    else:
        print("Thank you for using our service")

#%%
def datafiltering(df):
    df = df[['Date', 'Price']]
    df.rename(columns={'Date':'Date', 'Price':'Sales'}, inplace=True)
    df = df.groupby('Date').sum()
    # df = df[(df['Sales'] > 0) & (df['Sales'] < 28000)]
    df = df.fillna(0)
    return df

#%%
def forecast_sales(df):
    print(df.isnull().sum())
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    plt.plot(df)
    # plot(df)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Sales')
    plt.show()
    # df = df.resample('MS').mean()
    df.fillna(0, inplace=True)
    print(df)
    print(df['Sales'].shape)
    tsplot(df['Sales'], lags=100)
    plt.show()
    return df

#%%
def decompose(df):
    #decompose sales
    print(df)
    decomposition = sm.tsa.seasonal_decompose(df['Sales'], model='additive', extrapolate_trend='freq', period=2)
    print(decomposition.trend)
    fig = decomposition.plot()
    fig.show()
    plt.show()
    #test for stationarity
    print("Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(df['Sales'])[1])
    #make sales stationary
    print(df['Sales'].values)
    df['Sales_box'] = df.apply(lambda x: boxcox1p(df['Sales'], 0.15), axis=0)
    
    print("Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(df['Sales_box'])[1])

    #plot sales
    tsplot(df['Sales_box'], lags=50)

    #decompose sales
    decomposition = sm.tsa.seasonal_decompose(df['Sales_box'], model='additive', extrapolate_trend='freq', period=1)
    fig = decomposition.plot()
    fig.show()
    plt.show()
    return df

#%%
def arima(df):
    #ACF and PACF for sales
    plt.figure(figsize=(10,8))
    ax = plt.subplot(211)
    sm.graphics.tsa.plot_acf(df['Sales_box'].values.squeeze(), lags=50, ax=ax)
    ax = plt.subplot(212)
    sm.graphics.tsa.plot_pacf(df['Sales_box'], lags=50, ax=ax)
    plt.tight_layout()
    plt.show()

    #set parameters
    Qs = range(0,2)
    qs = range(0,2)
    Ps = range(0,2)
    ps = range(0,2)
    D = 1
    d = 1
    parameters = product(ps, qs, Ps, Qs)
    parameters_list = list(parameters)
    print(len(parameters_list))

    #find best parameters
    results = []
    best_aic = float("inf")
    print(results)
    print(best_aic)
    for param in parameters_list:
        try:
            model = sm.tsa.statespace.SARIMAX(df['Sales_box'], order=(param[0], d, param[1]),
                                            seasonal_order=(param[2], D, param[3], 12)).fit(disp=-1)
            results.append([param, model.aic])
            if model.aic < best_aic:
                best_model = model
                print(best_model)
                best_aic = model.aic
                best_param = param
                print(best_param)
        except Exception as err:
            print(err)
            # print("exception in the data")
    
    print(best_model.summary())
    return best_model
    # return {{"best_model": best_model, "best_param": best_param}}

#%%
def residuals(df, best_model):
    #plot residuals
    # plt.figure(figsize=(15,8))
    # plt.subplot(211)
    best_model.resid[13:].plot()
    plt.show()
    best_model.resid[13:].plot(kind='kde')
    plt.show()
    print("Dickey-Fuller test: p=%f" % sm.tsa.stattools.adfuller(best_model.resid[13:])[1])

    #forecast sales
    df['forecast'] = inv_boxcox(best_model.predict(start=0, end=161), 0.15)
    df[['Sales', 'forecast']].plot(figsize=(15, 6))
    plt.show()

    #forecast sales for next 12 months
    df['forecast'] = inv_boxcox(best_model.predict(start=0, end=100), 0.15)
    df[['Sales', 'forecast']].plot(figsize=(15, 6))
    plt.show()
    # return {{"df": df}}

#%%
def plot(df):
    plt.plot(df['Sales'])
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Sales')
    plt.show()

# %%
#monthwise sales prediction function
def monthwise_sales_prediction(best_model, month):
    month = pd.to_datetime(month)
    df['forecast'] = inv_boxcox(best_model.predict(start=0, end=month), 0.15)
    df[['Sales', 'forecast']].plot(figsize=(15, 6))
    plt.show()

#%%
def futuresales(best_model, df):
    from pandas.tseries.offsets import DateOffset
    future_dates = [df.index[-1] + DateOffset(months=x) for x in range(0,24)]
    future_datest_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)
    future_df = pd.concat([df, future_datest_df])
    future_df['forecast'] = inv_boxcox(best_model.predict(start=0, end=1000), 0.15)
    print(future_df)
    future_df[['Sales', 'forecast']].plot(figsize=(15, 6))
    plt.show()

#%%
def futuremonth(best_model, df):
    from pandas.tseries.offsets import DateOffset
    future_dates = [df.index[-1] + DateOffset(months=x) for x in range(0,2)]
    future_datest_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)
    future_df = pd.concat([df, future_datest_df])
    future_df['forecast'] = inv_boxcox(best_model.predict(start=0, end=100), 0.15)
    print(future_df)
    print("_"*50)
    print("Sales forecast for next month is: ", future_df['forecast'].iloc[-1])
    future_df[['Sales', 'forecast']].plot(figsize=(15, 6))
    plt.show()

#%%
def futurethreemonths(best_model, df):
    from pandas.tseries.offsets import DateOffset
    future_dates = [df.index[-1] + DateOffset(months=x) for x in range(0,4)]
    future_datest_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)
    future_df = pd.concat([df, future_datest_df])
    future_df['forecast'] = inv_boxcox(best_model.predict(start=0, end=100), 0.15)
    print(future_df)
    print("_"*50)
    print("Sales forecast for next 3 months is: ", future_df['forecast'].iloc[-1])
    future_df[['Sales', 'forecast']].plot(figsize=(15, 6))
    plt.show()

#%%
def futureoneday(best_model, df):
    from pandas.tseries.offsets import DateOffset
    future_dates = [df.index[-1] + DateOffset(months=x) for x in range(0,1)]
    future_datest_df = pd.DataFrame(index=future_dates[1:], columns=df.columns)
    future_df = pd.concat([df, future_datest_df])
    future_df['forecast'] = inv_boxcox(best_model.predict(start=0, end=100), 0.15)
    print(future_df)
    print("_"*50)
    print("Sales forecast for next 1 day is: ", future_df['forecast'].iloc[-1])
    future_df[['Sales', 'forecast']].plot(figsize=(15, 6))
    plt.show()

#%%
df = pd.read_csv("/home/drox/Documents/adeyelta/Adeyelta-model/data/guitarshop_sales_2022.csv")
df = datafiltering(df)
plot(df)
df = forecast_sales(df)
df = decompose(df)
best_model = arima(df)
residuals(df, best_model)
# month = input("enter a month to forecast: ")
# monthwise_sales_prediction(best_model, month)
futuresales(best_model, df)
futuremonth(best_model, df)
futurethreemonths(best_model, df)
futureoneday(best_model, df)

# %%
