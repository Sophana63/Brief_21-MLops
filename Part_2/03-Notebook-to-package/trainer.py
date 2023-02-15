import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import set_config
set_config(display='diagram')

from encoders import DistanceTransformer, TimeFeaturesEncoder

class Trainer:
    def __init__(self):
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

    def train(self, X_train, y_train):
        self.pipe.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test):
        y_pred = self.pipe.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print("RMSE: ", rmse)
        return rmse
