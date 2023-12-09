import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Download necessary NLTK data (do this once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt')
    
def preprocess_text(text):
    # Convert text to lowercase
    cleaned_text = text.lower()

    # Remove punctuation
    cleaned_text = cleaned_text.translate(str.maketrans('', '', string.punctuation))

    # Tokenization
    tokens = word_tokenize(cleaned_text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Join the cleaned tokens back into a string
    cleaned_text = ' '.join(lemmatized_tokens)
    return cleaned_text