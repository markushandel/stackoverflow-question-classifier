import json
import textstat
import nltk
from nltk import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
import string
import numpy as np
import re
from collections import Counter
nltk.download('punkt')

def extract_code(text):
    # Extracting code enclosed in <code> tags
    code_blocks = re.findall(r'<code>(.*?)</code>', text, flags=re.DOTALL)
    non_code_text = re.sub(r'<code>.*?</code>', '', text, flags=re.DOTALL)

    # Joining all code blocks into a single string
    combined_code = ' '.join(code_blocks)

    return combined_code, non_code_text


def analyze_text_quality(text):
    # Calculate readability scores
    readability = textstat.flesch_reading_ease(text)

    # Sentence and word complexity
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences) if sentences else 0
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    complex_words = sum(1 for word in words if textstat.syllable_count(word) > 3)

    return {
        "readability": readability,
        "avg_sentence_length": avg_sentence_length,
        "avg_word_length": avg_word_length,
        "complex_words": complex_words
    }

def clean_text(text):
    # Clean text
    cleaned_text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    cleaned_text = cleaned_text.lower().translate(str.maketrans('', '', string.punctuation))

    return cleaned_text


def process_code(code):
    # Basic metrics
    lines = code.split('\n')
    num_lines = len(lines)
    empty_lines = sum(1 for line in lines if line.strip() == '')
    comments = sum(1 for line in lines if line.strip().startswith('#') or line.strip().startswith('//'))

    # Advanced metrics
    functions = len(re.findall(r'\bdef\b|\bfunction\b', code))
    loops = len(re.findall(r'\bfor\b|\bwhile\b', code))
    conditionals = len(re.findall(r'\bif\b|\belse\b|\belif\b|\bswitch\b', code))

    # Language-specific features (example for Python and JavaScript)
    python_specific = len(re.findall(r'\bimport\b|\bfrom\b', code))
    javascript_specific = len(re.findall(r'\bconsole.log\b|\bdocument.getElementById\b', code))

    # Length-based metrics
    avg_line_length = sum(len(line) for line in lines) / num_lines if num_lines > 0 else 0
    max_line_length = max(len(line) for line in lines) if lines else 0

    # Code complexity (simplified version)
    unique_tokens = len(set(re.findall(r'\b\w+\b', code)))
    token_frequency = Counter(re.findall(r'\b\w+\b', code))

    return {
        "num_lines": num_lines,
        "empty_lines": empty_lines,
        "comments": comments,
        "functions": functions,
        "loops": loops,
        "conditionals": conditionals,
        "python_specific": python_specific,
        "javascript_specific": javascript_specific,
        "avg_line_length": avg_line_length,
        "max_line_length": max_line_length,
        "unique_tokens": unique_tokens,
        "token_frequency": token_frequency
    }


def preprocess_data(df):
    processed_data = []
    for index, row in df.iterrows():
        code, cleaned_body = extract_code(row['body'])
        code_features = process_code(code)
        text_metrics = analyze_text_quality(cleaned_body)

        processed_question = {
            'title': clean_text(row['title']),
            'body': clean_text(cleaned_body),
            'code': code,
            'tags': row['tags'],
            'has_code': int(bool(code)),
            'user_reputation': row['owner'].get('reputation', 0) if isinstance(row['owner'], dict) else 0,
            'label': row['closed_reason'],  # Use the class label from the file name
            'tag_amount': len(row['tags'])
        }

        processed_question.update(code_features)
        processed_question.update(text_metrics)

        processed_data.append(processed_question)

    # Convert the list of dictionaries to a DataFrame
    return pd.DataFrame(processed_data)


def process_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    return text.lower().translate(str.maketrans('', '', string.punctuation))


def vectorize_text(data, vectorizer):
    return vectorizer.fit_transform(data).toarray()


def encode_tags(data, encoder):
    reshaped_data = data.apply(lambda x: ','.join(x)).values.reshape(-1, 1)
    encoded_data = encoder.fit_transform(reshaped_data)
    return pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out())

