from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
    def fit(self,X,y=None):
        self.scaler.fit(X.select_dtypes(include=['number']))
        return self
    def transform(self,X):
        c = X.copy()
        numerical_cols = X.select_dtypes(include=['number']).columns
        c[numerical_cols] = self.scaler.transform(c[numerical_cols])
        return c 