from .base_data_processor import AbstractDataProcessor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class TfidfProcessor(AbstractDataProcessor):
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=2000)
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore')
        self.is_fitted = False

    def fit(self, df):
        # Fit the vectorizer and encoder only on training data
        self.tfidf_vectorizer.fit(df['title'] + ' ' + df['body'])
        self.onehot_encoder.fit(df['tags'].apply(lambda x: ','.join(x)).values.reshape(-1, 1))
        self.is_fitted = True

    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError("DataProcessor must be fitted before calling transform")

        # Transforming text columns
        title_vectorized = self.tfidf_vectorizer.transform(df['title']).toarray()
        body_vectorized = self.tfidf_vectorizer.transform(df['body']).toarray()
        encoded_tags = self.onehot_encoder.transform(df['tags'].apply(lambda x: ','.join(x)).values.reshape(-1, 1)).toarray()

        # Selecting non-string columns (numeric or boolean types)
        non_string_columns = df.select_dtypes(include=['number', 'bool']).values

        # Concatenating all features
        return np.hstack((title_vectorized, body_vectorized, encoded_tags, non_string_columns))