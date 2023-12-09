from .base_data_processor import AbstractDataProcessor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class BowDataProcessor(AbstractDataProcessor):
    def __init__(self):
        super().__init__()
        self.count_vectorizer = CountVectorizer(max_features=2000)
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore')

    def fit(self, df):
        self.count_vectorizer.fit(df['title'] + ' ' + df['body'])
        self.onehot_encoder.fit(df['tags'].apply(lambda x: ','.join(x)).values.reshape(-1, 1))
        self.is_fitted = True

    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError("DataProcessor must be fitted before calling transform")

        title_vectorized = self.count_vectorizer.transform(df['title']).toarray()
        body_vectorized = self.count_vectorizer.transform(df['body']).toarray()
        encoded_tags = self.onehot_encoder.transform(df['tags'].apply(lambda x: ','.join(x)).values.reshape(-1, 1)).toarray()

        non_string_columns = df.select_dtypes(include=['number', 'bool']).values

        return np.hstack((title_vectorized, body_vectorized, encoded_tags, non_string_columns))
