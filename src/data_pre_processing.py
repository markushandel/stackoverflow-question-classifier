import json
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import string
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    return text.lower().translate(str.maketrans('', '', string.punctuation))


def preprocess_data(data):
    processed_data = []
    for question in data:
        processed_question = {
            'title': clean_text(question['title']),
            'body': clean_text(question['body']),  # Change made here
            'tags': question['tags'],
            'user_reputation': question['owner'].get('reputation', None),
        }
        processed_data.append(processed_question)
    return processed_data

def vectorize_text(data, vectorizer):
    return vectorizer.fit_transform(data).toarray()

def encode_tags(data, encoder):
    reshaped_data = data.apply(lambda x: ','.join(x)).values.reshape(-1, 1)
    encoded_data = encoder.fit_transform(reshaped_data)
    return pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out())

def main():
    folder_path = "data/raw"  # Your data folder path
    all_data = []
    labels = []  # To store labels

    for file_name in os.listdir(folder_path):
        class_label = file_name.split('-')[0]  # Assuming the file name starts with the class label
        file_path = os.path.join(folder_path, file_name)
        data = load_data(file_path)
        preprocessed_data = preprocess_data(data)

        for question in preprocessed_data:
            question['label'] = class_label  # Assign label to each question

        all_data.extend(preprocessed_data)
        labels.extend([class_label] * len(data))  # Extend labels list

    df = pd.DataFrame(all_data)

    # Vectorize textual data
    tfidf_vectorizer = TfidfVectorizer(max_features=100)  # Adjust parameters as needed

    # Vectorize 'title' and 'body' separately and convert to dense arrays
    title_vectorized = tfidf_vectorizer.fit_transform(df['title']).toarray()
    body_vectorized = tfidf_vectorizer.fit_transform(df['body']).toarray()

    # Encode tags
    onehot_encoder = OneHotEncoder(handle_unknown='ignore')
    encoded_tags = onehot_encoder.fit_transform(df['tags'].apply(lambda x: ','.join(x)).values.reshape(-1, 1)).toarray()

    # Combine all features into one array (ensure all have the same number of rows)
    features = np.hstack((title_vectorized, body_vectorized, encoded_tags))

    print("featueres", features.shape, len(labels))

    # Split dataset into train and test
    train_features, test_features, y_train, y_test = train_test_split(features, df['label'], test_size=0.2)

    # Train the model
    model = LogisticRegression(max_iter=1000)  # Adjust parameters as needed
    model.fit(train_features, y_train)

    # Make predictions
    train_predictions = model.predict(train_features)
    test_predictions = model.predict(test_features)

    # Calculate and print metrics
    print("Train Metrics:")
    print(classification_report(y_train, train_predictions))
    print("Accuracy:", accuracy_score(y_train, train_predictions))

    print("\nTest Metrics:")
    print(classification_report(y_test, test_predictions))
    print("Accuracy:", accuracy_score(y_test, test_predictions))

if __name__ == "__main__":
    main()
