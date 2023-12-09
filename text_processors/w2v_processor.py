from .base_data_processor import AbstractDataProcessor
from sklearn.preprocessing import OneHotEncoder
from gensim.models import Word2Vec
import numpy as np


class Word2VecDataProcessor(AbstractDataProcessor):
    def __init__(self):
        super().__init__()
        self.model = Word2Vec()
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore')

    def fit(self, df):
        self.onehot_encoder.fit(df['tags'].apply(lambda x: ','.join(x)).values.reshape(-1, 1))
        # Train or fine-tune the Word2Vec model here if needed
        # This step depends on whether you're using a pre-trained model
        self.is_fitted = True

    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError("DataProcessor must be fitted before calling transform")

        # Function to create embeddings for a single text entry
        def create_embeddings(text):
            words = text.split()
            vectors = [self.model.wv[word] for word in words if word in self.model.wv]
            if len(vectors) > 0:
                return np.mean(vectors, axis=0)
            else:
                return np.zeros(self.model.vector_size)

        # Create embeddings for each text entry
        title_embeddings = np.array([create_embeddings(text) for text in df['title']])
        body_embeddings = np.array([create_embeddings(text) for text in df['body']])
        
        # One-hot encode the tags
        encoded_tags = self.onehot_encoder.transform(df['tags'].apply(lambda x: ','.join(x)).values.reshape(-1, 1)).toarray()

        # Get non-string columns
        non_string_columns = df.select_dtypes(include=['number', 'bool']).values

        # Combine all features
        return np.hstack((title_embeddings, body_embeddings, encoded_tags, non_string_columns))