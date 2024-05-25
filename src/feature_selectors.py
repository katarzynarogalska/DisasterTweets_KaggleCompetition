from sklearn.base import BaseEstimator,TransformerMixin


class FirstFeatureSelector(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        recursive_elimination_features = ['count_caps_lock','count_exclamation_mark','positive','polarity','count_links','count_mentions','count_nouns']
        return X[recursive_elimination_features]
    

class SecondFeatureSelector(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        random_forest_features = ['count_words','count_punctuation','polarity','count_stopwords','subjectivity','count_nouns']
        return X[random_forest_features]
    
class ThirdFeatureSelector(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        KBest_features = ['count_words', 'count_punctuation', 'count_links', 'count_nouns','polarity']
        return X[KBest_features]