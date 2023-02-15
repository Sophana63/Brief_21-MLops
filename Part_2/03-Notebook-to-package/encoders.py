import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

import utils

class Trainer :
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """

    def __init__(self,
                data_link = "../data/train.csv"):
        self.data_link = data_link
        self.dist_pipe = Pipeline([
            ('dist', DistanceTransformer()),
            ('scaler', StandardScaler())
        ])

        self.time_pipe = Pipeline([
            ('time', TimeFeaturesEncoder()),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preproc_pipe = FeatureUnion([
            ('dist_pipe', self.dist_pipe),
            ('time_pipe', self.time_pipe)
        ])

        self.model = LinearRegression()
        self.pipe = Pipeline([
            ('preprocessor', self.preproc_pipe),
            ('regressor', self.model)
        ])


    def get_clean_data_train(self, nrows=10_000, test=False, test_size=0.2):
        df = pd.read_csv(self.data_link, nrows=nrows)
        df = df.dropna()

        # remove the outliers
        if not test:
            df = df[(df.fare_amount >= 2.5) & (df.fare_amount <= 150)]

        y = df["fare_amount"]
        X = df.drop("fare_amount", axis=1)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15)
        return X_train, X_val, y_train, y_val

    def train(self, X_train, y_train):
        self.pipe.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test):
        y_pred = self.pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        score = self.pipe.score(X_test, y_test)
        print("-----------------------------------------------------")
        print ("Predict: ", y_pred)
        print("-----------------------------------------------------")
        print("RMSE: ", rmse)
        print("-----------------------------------------------------")
        return rmse, score   

    

# create a TimeFeaturesEncoder
class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):
    """
        Extracts the day of week (dow), the hour, the month and the year from a time column.
        Returns a copy of the DataFrame X with only four columns: 'dow', 'hour', 'month', 'year'.
    """

    def __init__(self, time_column='pickup_datetime'):
        self.time_column = time_column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_[self.time_column] = pd.to_datetime(X_[self.time_column])
        X_['dow'] = X_[self.time_column].dt.dayofweek
        X_['hour'] = X_[self.time_column].dt.hour
        X_['month'] = X_[self.time_column].dt.month
        X_['year'] = X_[self.time_column].dt.year
        return X_[['dow', 'hour', 'month', 'year']]

class DistanceTransformer(BaseEstimator, TransformerMixin):
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """

    def __init__(self,
                 start_lat="pickup_latitude",
                 start_lon="pickup_longitude",
                 end_lat="dropoff_latitude",
                 end_lon="dropoff_longitude"):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.end_lat = end_lat
        self.end_lon = end_lon

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        X_['distance'] = utils.haversine_vectorized(X_,
                                              start_lat=self.start_lat,
                                              start_lon=self.start_lon,
                                              end_lat=self.end_lat,
                                              end_lon=self.end_lon)
        return X_[['distance']]