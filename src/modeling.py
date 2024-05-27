from keras import layers
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

class GensimVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_type='cbow', size=100, window=5, min_count=1, workers=4):
        self.model_type = model_type
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers

    def fit(self, X, y=None):
        model_type = 0 if self.model_type == 'cbow' else 1
        self.model = Word2Vec(X, vector_size=self.size, window=self.window, min_count=self.min_count, workers=self.workers, sg=model_type)
        return self

    def transform(self, X):
        return np.array([np.mean([self.model.wv[word] for word in sentence if word in self.model.wv] or [np.zeros(self.size)], axis=0) for sentence in X])



vectorizers = {
    'Tfidf': TfidfVectorizer(),
    'Count': CountVectorizer(),
    'Skipgram': GensimVectorizer(model_type='skipgram'),
    'CBow': GensimVectorizer(model_type='cbow')
}

# Define models
models = {
    'MultinomialNB': MultinomialNB(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
    
}

# Define parameter grids for models
param_grid = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'SVM': {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'Logistic Regression': {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'MultinomialNB': {
        'alpha': [0.01, 0.1, 1, 10, 100]
    }
}

# Vectorizer-specific parameters
vectorizer_params = {
    'max_features': [1000, 2000, 3000],
    'ngram_range': [(1, 1), (1, 2)]
}

class GensimVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_type='cbow', size=100, window=5, min_count=1, workers=4):
        self.model_type = model_type
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers

    def fit(self, X, y=None):
        model_type = 0 if self.model_type == 'cbow' else 1
        self.model = Word2Vec(X, vector_size=self.size, window=self.window, min_count=self.min_count, workers=self.workers, sg=model_type)
        return self

    def transform(self, X):
        return np.array([np.mean([self.model.wv[word] for word in sentence if word in self.model.wv] or [np.zeros(self.size)], axis=0) for sentence in X])

def transform_data(vectorizer, X_train, X_test):
    X_train_transformed = vectorizer.fit_transform(X_train['processed_text_str'])
    X_test_transformed = vectorizer.transform(X_test['processed_text_str'])
    return X_train_transformed, X_test_transformed

def run_model(vectorizer, X_train_transformed, X_test_transformed,y_train, y_test, model, param_dist):
    random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=5, cv=3, scoring='f1', n_jobs=-1)
    random_search.fit(X_train_transformed, y_train)
    
    best_estimator = random_search.best_estimator_
    predictions = best_estimator.predict(X_test_transformed)
    
    # print(f"Classes with no predicted samples: {set(y_test) - set(predictions)}")

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    
    return {
        'Accuracy': report['accuracy'],
        'Precision': report['weighted avg']['precision'],
        'Recall': report['weighted avg']['recall'],
        'F1-Score': report['weighted avg']['f1-score']
    }

def run_vectorizer_model(vectorizers, X_train, X_test, y_train, y_test, models, param_grid):
    results = []
    pbar = tqdm(total=len(vectorizers) * len(vectorizer_params['max_features']) * len(vectorizer_params['ngram_range']) * len(models))
    
    for vec_name, vectorizer in vectorizers.items():
        if vec_name in ['Tfidf', 'Count']:
            for max_feat in vectorizer_params['max_features']:
                for ngram in vectorizer_params['ngram_range']:
                    pbar.set_description(f"Processing {vec_name} vectorizer (max_features={max_feat}, ngram_range={ngram})")
                    pbar.update(1)
                    
                    vectorizer.set_params(max_features=max_feat, ngram_range=ngram)
                    X_train_transformed, X_test_transformed = transform_data(vectorizer, X_train, X_test)
                    
                    for model_name, model in models.items():
                        params = param_grid[model_name]
                        param_dist = {key: [value] if isinstance(value, int) else value for key, value in params.items()}
                        
                        if model_name == 'MultinomialNB' and 'Tfidf' not in vec_name:
                            continue
                        
                        best_params = run_model(vectorizer, X_train_transformed, X_test_transformed,y_train, y_test, model, param_dist)
                        
                        results.append({
                            'Vectorizer': vec_name,
                            'Model': model_name,
                            'Max Features': max_feat,
                            'N-gram Range': ngram,
                            **best_params
                        })
                        pbar.update(1)
        else:
            pbar.set_description(f"Processing {vec_name} vectorizer")
            
            X_train_transformed, X_test_transformed = transform_data(vectorizer, X_train, X_test)
            
            for model_name, model in models.items():
                params = param_grid[model_name]
                param_dist = {key: [value] if isinstance(value, int) else value for key, value in params.items()}
                
                if model_name == 'MultinomialNB':
                    continue
                
                best_params = run_model(vectorizer, X_train_transformed, X_test_transformed, y_train, y_test, model, param_dist)
                
                results.append({
                    'Vectorizer': vec_name,
                    'Model': model_name,
                    **best_params
                })
                pbar.update(1)
    
    pbar.close()
    return results

# This is for keras Model
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions