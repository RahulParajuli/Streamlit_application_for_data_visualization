from models.regression import *
from readfile.readfile import *
from salesforecasting.salesforcasting import *
from salesforecasting.LSTMforecasting import *
from salesforecasting.linearmodel import *

def main():
    #read the data
    df = readfile()
    df = datafiltering(df)
    #preprocess the data
    # df = preprocess(df)
    #sales forecasting
    forecast = sales_forecast(df)
    #sales prediction using linear regression
    print("forecast using linear regression")
    sales_using_linear(df)
    print("forecasting using LSTM model")
    sales_using_LSTM(df)

if __name__ == "__main__":
    main()