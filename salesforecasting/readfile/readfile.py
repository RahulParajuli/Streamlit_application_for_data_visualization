#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

def main(df):
    df = readfile(df)
    print(df)


if __name__ == "__main__":
    main()

