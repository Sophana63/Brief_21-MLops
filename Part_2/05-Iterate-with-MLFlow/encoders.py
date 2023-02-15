from functools import cached_property
from matplotlib.widgets import Lasso
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

import mlflow
from mlflow.tracking import MlflowClient
from memoized_property import memoized_property
from sklearn.tree import DecisionTreeRegressor
import utils

import joblib

class Trainer :
    """
        Computes the haversine distance between two GPS points.
        Returns a copy of the DataFrame X with only one column: 'distance'.
    """

    def __init__(self,
                data_link = "../data/train.csv",
                experiment_name = "FR.63.Sophana63.PipelineModel.V1.0",
                tracking_uri = "file:///C:/Users/Administrateur/Projets/Brief_21-MLops/Part_2/05-Iterate-with-MLFlow/mlruns",
                ):
        self.data_link = data_link
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
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

        self.estimators = [
            ("Linear Regression", LinearRegression()),
            ("Ridge Regression", Ridge())
            ("Decision Tree", DecisionTreeRegressor())
            ("Random Forest", RandomForestRegressor())
            ]

        self.scores = []


    def get_clean_data_train(self, nrows=10_000, test=False, test_size=0.3):
        df = pd.read_csv(self.data_link, nrows=nrows)
        df = df.dropna()

        # remove the outliers
        if not test:
            df = df[(df.fare_amount >= 2.5) & (df.fare_amount <= 150)]

        y = df["fare_amount"]
        X = df.drop("fare_amount", axis=1)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size)
        return X_train, X_val, y_train, y_val, X, y

    def train(self, X_train, y_train):
        self.pipe.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test):
        y_pred = self.pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print("-----------------------------------------------------")
        print ("Predict: ", y_pred)
        print("-----------------------------------------------------")
        print("RMSE: ", rmse)
        print("-----------------------------------------------------")
        return rmse  

    def evaluate_estimators(self, X_train, y_train, X_test, y_test, X, y):
        results = { 
            "name" : [],
            "score" : [],
            "RMSE" : [],
            "MSE" : [],
            "MAE" : []
        }
        for name, estimator in self.estimators:
            self.pipe_estimators = Pipeline([
                ('preprocessor', self.preproc_pipe),
                (name, estimator)
            ])            
            self.pipe_estimators.fit(X_train, y_train)
            cv_score_mse = cross_val_score(self.pipe_estimators, X, y, scoring="neg_mean_squared_error", cv=5)
            cv_score_mae = cross_val_score(self.pipe_estimators, X, y, scoring="neg_mean_absolute_error", cv=5)
            y_pred = self.pipe_estimators.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            score = self.pipe_estimators.score(X_test, y_test)
            self.scores.append(score)

            mse = np.mean(np.abs(cv_score_mse))
            mae = np.mean(np.abs(cv_score_mae))
            print(f"{name} - Score: {score} - RMSE: {rmse} - Mean MSE: {mse} - Mean MAE: {mae}")

            results["name"].append(name)
            results["score"].append(score)
            results["RMSE"].append(rmse)
            results["MSE"].append(mse)
            results["MAE"].append(mae)

        return results

    @cached_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(self.tracking_uri)
        return MlflowClient()

    @cached_property
    def mlflow_experiment_id(self):
        try:
           return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id
    
    @cached_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value) 

    def save_model(self, model_train):
        joblib.dump(model_train, "model.joblib")
        pass


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