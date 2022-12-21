#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import datetime as dt

#%%
#sales using KNN regression
def sales_using_KNN(df):
    df = df[["Created at", "Paid Price"]]
    df = df.rename(columns={"Created at": "Date", "Paid Price": "Sales"})
    print(df.columns)
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
    return df
#%%
def KNN(df):
    df['Sales'] = df['Sales'].astype(int)
    df['Date'] = df.index
    df['Date'] = df['Date'].map(dt.datetime.toordinal)
    X = df[['Date']]
    y = df['Sales']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regressor = KNeighborsRegressor(n_neighbors=2)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    print(df2)
    df2.plot(kind='bar',figsize=(10,8))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.show()
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    print('R2 score:', metrics.r2_score(y_test, y_pred))
    print('Accuracy:', regressor.score(X_test, y_test))
    print('score:', regressor.score(X_test, y_test))
    plt.scatter(X_test, y_test,  color='gray')
    plt.plot(X_test, y_pred, color='red', linewidth=2)
    plt.show
    # return {{"Mean Absolute Error": metrics.mean_absolute_error(y_test, y_pred), "Mean Squared Error": metrics.mean_squared_error(y_test, y_pred), "Root Mean Squared Error": np.sqrt(metrics.mean_squared_error(y_test, y_pred)), "R2 score": metrics.r2_score(y_test, y_pred), "Accuracy": regressor.score(X_test, y_test), "score": regressor.score(X_test, y_test)}}
    
#%%
def readfile():
    filename = input("enter a file path: ")
    if filename.endswith('.csv'):
        df = pd.read_csv(filename,sep=';', header=0)
        print("you have entered a CSV file")
    elif filename.endswith('.xlsx'):
        df = pd.read_excel(filename)
        print("you have entered a XLSX file")
    else:
        raise ValueError('Unknown file type')
    return df


#%%
import pandas as pd
df = readfile()
df = sales_using_KNN(df)
KNN(df)

# %%
