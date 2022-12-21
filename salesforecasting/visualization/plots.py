import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot(df):
    df.plot(kind='bar',figsize=(10,10))
    plt.show()

def histogram(df):
    df.hist()
    plt.show()

def boxplot(df):
    df.boxplot()
    plt.show()

def scatterplot(df):
    df.plot(kind='scatter', x='x', y='y')
    plt.show()

def lineplot(df):
    df.plot(kind='line', x='x', y='y')
    plt.show()

def piechart(df):
    df.plot(kind='pie', y='y')
    plt.show()

def barplot(df):
    df.plot(kind='bar', x='x', y='y')
    plt.show()

def heatmap(df):
    df.plot(kind='heatmap', x='x', y='y')
    plt.show()

def main():
    df = pd.DataFrame(np.random.randint(0,100,size=(100, 4)), columns=list('ABCD'))
    plot(df)
    histogram(df)
    boxplot(df)
    scatterplot(df)
    lineplot(df)
    piechart(df)
    barplot(df)
    heatmap(df)

if __name__ == "__main__":
    main()
    