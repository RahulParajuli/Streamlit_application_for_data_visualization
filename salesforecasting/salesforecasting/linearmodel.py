#%%
def libraries():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import sklearn
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    import datetime as dt
    from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error

#%%
#sales prediction using linear regression
def dateTime(df):
    print(df)
    import matplotlib.pyplot as plt
    import datetime as dt
    from sklearn.model_selection import train_test_split
    df = df[['Created at', 'Paid Price']]
    df.rename(columns={'Created at':'Date', 'Paid Price':'Sales'}, inplace=True)
    # df = df[(df['Sales'] > 0) & (df['Sales'] < 100000)]
    df = df.fillna(0)
    df = df.set_index('Date')
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.resample('MS').mean()
    df.fillna(0, inplace=True)
    print(df)
    plt.plot(df)
    plt.xlabel('Date')
    plt.ylabel('Sales')
    plt.title('Sales')
    plt.show()
    df['Sales'] = df['Sales'].astype(int)
    df['Date'] = df.index
    df['Date'] = df['Date'].map(dt.datetime.toordinal)
    X = df[['Date']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test

#%%
def linearmodel(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    from sklearn import metrics
    import numpy as np
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    df2.to_csv('/home/drox/Documents/adeyelta/Adeyelta-model/salesforecasting/extractedModel/linearModel.csv')
    print(df2)
    
    ax = df2[['Actual']].plot(figsize=(15,5))
    df2['Predicted'].plot(ax=ax, style='-')
    ax.set_title("Raw data and prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    plt.show()

    df2.plot(kind='bar',figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R2 score:', metrics.r2_score(y_test, y_pred))
    print('Accuracy:', regressor.score(X_test, y_test))
    print('intercept:', regressor.intercept_)
    print('slope:', regressor.coef_)
    print('score:', regressor.score(X_test, y_test))
    plt.scatter(X_test, y_test,  color='gray')
    plt.plot(X_test, y_pred, color='red', linewidth=2)
    plt.show()
    # return {{"Mean Absolute Error": metrics.mean_absolute_error(y_test, y_pred), "Mean Squared Error": metrics.mean_squared_error(y_test, y_pred), "Root Mean Squared Error": np.sqrt(metrics.mean_squared_error(y_test, y_pred)), "R2 score": metrics.r2_score(y_test, y_pred), "Accuracy": regressor.score(X_test, y_test), "intercept": regressor.intercept_, "slope": regressor.coef_, "score": regressor.score(X_test, y_test)}}


#%%
import pandas as pd
libraries()
df = pd.read_csv('/home/drox/Documents/adeyelta/Adeyelta-model/data/Guitar_shop.csv', sep=";")
X_train, X_test, y_train, y_test = dateTime(df)
linearmodel(X_train, X_test, y_train, y_test)


# %%
