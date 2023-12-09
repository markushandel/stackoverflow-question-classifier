from .base_data_processor import AbstractDataProcessor
from sklearn.preprocessing import OneHotEncoder
from transformers import BertModel, BertTokenizer
import numpy as np


class BertDataProcessor(AbstractDataProcessor):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')  # Example model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.onehot_encoder = OneHotEncoder(handle_unknown='ignore')

    def fit(self, df):
        # Fit the one-hot encoder with the 'tags' column
        self.onehot_encoder.fit(df['tags'].apply(lambda x: ','.join(x)).values.reshape(-1, 1))
        
        # Set the is_fitted flag to True
        self.is_fitted = True

    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError("DataProcessor must be fitted before calling transform")

        # Tokenize and create BERT embeddings
        # Again, this is a simplified example
        title_embeddings = self.create_bert_embeddings(df['title'])
        body_embeddings = self.create_bert_embeddings(df['body'])
        encoded_tags = self.onehot_encoder.transform(df['tags'].apply(lambda x: ','.join(x)).values.reshape(-1, 1)).toarray()

        non_string_columns = df.select_dtypes(include=['number', 'bool']).values

        return np.hstack((title_embeddings, body_embeddings, encoded_tags, non_string_columns))

    def create_bert_embeddings(self, texts):
        # Function to tokenize texts and get BERT embeddings
        # Simplified for example purposes
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            outputs = self.model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].detach().numpy())
        return np.vstack(embeddings)
