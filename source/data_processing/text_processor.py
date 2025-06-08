"""
Text processing module for AMT.
Contains functions for processing text descriptions and generating embeddings.
"""

import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from typing import List, Dict, Any
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from collections import Counter

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model for music-specific NLP
try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Downloading spaCy model...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm')

# Music-specific keywords and patterns
MUSIC_GENRES = {
    'classical', 'jazz', 'rock', 'pop', 'blues', 'folk', 'electronic', 
    'hip hop', 'r&b', 'country', 'metal', 'reggae', 'latin'
}

MUSIC_INSTRUMENTS = {
    'piano', 'guitar', 'violin', 'drums', 'bass', 'saxophone', 'trumpet',
    'flute', 'clarinet', 'cello', 'viola', 'organ', 'synthesizer'
}

MUSIC_EMOTIONS = {
    'happy', 'sad', 'energetic', 'calm', 'melancholic', 'joyful', 'peaceful',
    'dramatic', 'romantic', 'mysterious', 'intense', 'relaxing'
}

def clean_text(text: str) -> str:
    """
    Clean and preprocess input text.
    Args:
        text: Input text to clean
    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-zA-Z\s.,!?]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_music_keywords(text: str) -> Dict[str, List[str]]:
    """
    Extract music-specific keywords from text.
    Args:
        text: Input text
    Returns:
        Dictionary containing different types of music keywords
    """
    doc = nlp(text.lower())
    words = [token.text for token in doc if not token.is_stop and token.is_alpha]
    
    # Extract different types of keywords
    keywords = {
        'genres': [word for word in words if word in MUSIC_GENRES],
        'instruments': [word for word in words if word in MUSIC_INSTRUMENTS],
        'emotions': [word for word in words if word in MUSIC_EMOTIONS]
    }
    
    return keywords

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords from text using TF-IDF.
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to extract
    Returns:
        List of keywords
    """
    # Clean text
    cleaned_text = clean_text(text)
    
    # Tokenize and remove stopwords
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(cleaned_text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(max_features=max_keywords)
    try:
        tfidf_matrix = vectorizer.fit_transform([cleaned_text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top keywords
        tfidf_scores = tfidf_matrix.toarray()[0]
        keyword_indices = np.argsort(tfidf_scores)[-max_keywords:]
        keywords = [feature_names[i] for i in keyword_indices]
        
        return keywords
    except:
        return []

def get_text_features(text: str) -> Dict[str, Any]:
    """
    Extract various features from text.
    Args:
        text: Input text
    Returns:
        Dictionary containing text features
    """
    # Clean text
    cleaned_text = clean_text(text)
    
    # Tokenize
    tokens = word_tokenize(cleaned_text)
    
    # Get music-specific keywords
    music_keywords = extract_music_keywords(text)
    
    # Calculate basic statistics
    features = {
        "word_count": len(tokens),
        "unique_words": len(set(tokens)),
        "avg_word_length": np.mean([len(word) for word in tokens]) if tokens else 0,
        "keywords": extract_keywords(text),
        "music_genres": music_keywords['genres'],
        "music_instruments": music_keywords['instruments'],
        "music_emotions": music_keywords['emotions']
    }
    
    return features

def process_text_descriptions(text_list: List[str]) -> Dict[str, Any]:
    """
    Process a list of text descriptions.
    Args:
        text_list: List of text descriptions
    Returns:
        Dictionary containing processed data and statistics
    """
    processed_data = []
    total_words = 0
    total_unique_words = set()
    genre_counter = Counter()
    instrument_counter = Counter()
    emotion_counter = Counter()
    
    for text in text_list:
        # Clean and process text
        cleaned_text = clean_text(text)
        features = get_text_features(text)
        
        # Update statistics
        total_words += features["word_count"]
        total_unique_words.update(word_tokenize(cleaned_text))
        
        # Update music-specific counters
        genre_counter.update(features["music_genres"])
        instrument_counter.update(features["music_instruments"])
        emotion_counter.update(features["music_emotions"])
        
        processed_data.append({
            "original_text": text,
            "cleaned_text": cleaned_text,
            "features": features
        })
    
    return {
        "processed_data": processed_data,
        "statistics": {
            "total_descriptions": len(text_list),
            "total_words": total_words,
            "total_unique_words": len(total_unique_words),
            "avg_words_per_description": total_words / len(text_list) if text_list else 0,
            "common_genres": genre_counter.most_common(5),
            "common_instruments": instrument_counter.most_common(5),
            "common_emotions": emotion_counter.most_common(5)
        }
    }

def get_bert_embeddings(text_list: List[str], model_name: str = 'bert-base-uncased') -> List[np.ndarray]:
    """
    Generate BERT embeddings for a list of texts.
    Args:
        text_list: List of input texts
        model_name: Name of BERT model to use
    Returns:
        List of embeddings (numpy arrays)
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()

    embeddings_list = []
    with torch.no_grad():
        for text in text_list:
            cleaned_text = clean_text(text)
            inputs = tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            outputs = model(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings_list.append(cls_embedding)
    
    return embeddings_list 