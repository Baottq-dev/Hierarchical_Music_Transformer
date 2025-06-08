"""
Text processing module for AMT.
Contains functions for processing text descriptions from Wikipedia.
"""

import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from typing import List, Dict, Any
import re
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import spacy
import requests
from bs4 import BeautifulSoup

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model
nlp = spacy.load('en_core_web_sm')

# Initialize BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Constants for music-specific keywords
MUSIC_GENRES = {
    'rock', 'pop', 'jazz', 'classical', 'electronic', 'hip hop', 'r&b', 'blues',
    'country', 'folk', 'metal', 'punk', 'reggae', 'soul', 'funk', 'disco'
}

MUSIC_INSTRUMENTS = {
    'piano', 'guitar', 'drums', 'bass', 'violin', 'saxophone', 'trumpet',
    'flute', 'clarinet', 'cello', 'viola', 'trombone', 'organ', 'synth'
}

MUSIC_EMOTIONS = {
    'happy', 'sad', 'energetic', 'calm', 'angry', 'peaceful', 'melancholic',
    'joyful', 'dark', 'bright', 'intense', 'soft', 'loud', 'gentle'
}

def clean_text(text: str) -> str:
    """
    Clean and preprocess text.
    Args:
        text: Input text
    Returns:
        Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def extract_music_keywords(text: str) -> Dict[str, List[str]]:
    """
    Extract music-specific keywords from text.
    Args:
        text: Input text
    Returns:
        Dictionary of extracted keywords by category
    """
    words = word_tokenize(text.lower())
    
    genres = [word for word in words if word in MUSIC_GENRES]
    instruments = [word for word in words if word in MUSIC_INSTRUMENTS]
    emotions = [word for word in words if word in MUSIC_EMOTIONS]
    
    return {
        "genres": genres,
        "instruments": instruments,
        "emotions": emotions
    }

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    Extract keywords using TF-IDF.
    Args:
        text: Input text
        max_keywords: Maximum number of keywords to extract
    Returns:
        List of extracted keywords
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=max_keywords,
            stop_words='english'
        )
        
        # Fit and transform text
        tfidf_matrix = vectorizer.fit_transform([text])
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Get top keywords
        keywords = []
        for i in range(len(feature_names)):
            if tfidf_matrix[0, i] > 0:
                keywords.append(feature_names[i])
        
        return keywords
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def get_text_features(text: str) -> Dict[str, Any]:
    """
    Extract various features from text.
    Args:
        text: Input text
    Returns:
        Dictionary of text features
    """
    # Clean text
    cleaned_text = clean_text(text)
    
    # Get music-specific keywords
    music_keywords = extract_music_keywords(cleaned_text)
    
    # Get general keywords
    keywords = extract_keywords(cleaned_text)
    
    # Get word count and unique words
    words = word_tokenize(cleaned_text)
    word_count = len(words)
    unique_words = len(set(words))
    
    return {
        "word_count": word_count,
        "unique_words": unique_words,
        "music_genres": music_keywords["genres"],
        "music_instruments": music_keywords["instruments"],
        "music_emotions": music_keywords["emotions"],
        "keywords": keywords
    }

def get_bert_embedding(text: str) -> np.ndarray:
    """
    Generate BERT embedding for text.
    Args:
        text: Input text
    Returns:
        BERT embedding vector
    """
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use [CLS] token embedding
    embedding = outputs.last_hidden_state[0, 0, :].numpy()
    
    return embedding

def scrape_wikipedia(artist: str, song: str) -> str:
    """
    Scrape Wikipedia for song information.
    Args:
        artist: Artist name
        song: Song title
    Returns:
        Scraped text description
    """
    try:
        # Construct search query
        query = f"{artist} {song}"
        url = f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}"
        
        # Get page content
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract main content
        content = soup.find('div', {'class': 'mw-parser-output'})
        if content:
            # Get first paragraph
            paragraphs = content.find_all('p')
            for p in paragraphs:
                if p.text.strip():
                    return p.text.strip()
        
        return ""
    except Exception as e:
        print(f"Error scraping Wikipedia: {e}")
        return ""

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

def create_training_examples(midi_file: str, text_description: str) -> Dict[str, Any]:
    """
    Create training example from MIDI file and text description.
    Args:
        midi_file: Path to MIDI file
        text_description: Text description
    Returns:
        Dictionary containing training example
    """
    # Process text
    text_features = get_text_features(text_description)
    text_embedding = get_bert_embedding(text_description)
    
    # Process MIDI
    from .midi_processor import midi_to_event_sequence
    event_sequence = midi_to_event_sequence(midi_file)
    
    # Create training example
    return {
        "midi_file": midi_file,
        "text_description": text_description,
        "text_features": text_features,
        "text_embedding": text_embedding.tolist(),
        "event_sequence": event_sequence
    } 