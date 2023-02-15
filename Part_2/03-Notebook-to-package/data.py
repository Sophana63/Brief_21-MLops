import pandas as pd

def get_data(data='../data/train.csv', nrows=10_000):
    df = pd.read_csv(data, nrows=nrows)
    return df

def clean_data(df, test=False):
    '''returns a DataFrame without outliers and missing values'''
    df = df.dropna()

    # remove the outliers
    if not test:
        df = df[(df.fare_amount >= 2.5) & (df.fare_amount <= 150)]
    return df