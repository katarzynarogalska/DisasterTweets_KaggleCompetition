from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
    def fit(self,X,y=None):
        self.scaler.fit(X.select_dtypes(include=['number']))
        return self
    def transform(self,X):
        numerical_cols = X.select_dtypes(include=['number']).columns
        X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        return X
    def set_output(self, *args, **kwargs):
        return self