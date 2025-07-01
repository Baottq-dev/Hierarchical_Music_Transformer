"""
Text Processor - Processes text descriptions for training
"""

import re
import json
import numpy as np
from typing import List, Dict, Any, Optional
from transformers import BertTokenizer, BertModel
import torch
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class TextProcessor:
    """Processes text descriptions for training."""
    
    def __init__(self, 
                 max_length: int = 512,
                 use_bert: bool = True,
                 use_spacy: bool = True):
        self.max_length = max_length
        self.use_bert = use_bert
        self.use_spacy = use_spacy
        
        # Initialize BERT
        if use_bert:
            try:
                self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.bert_model = BertModel.from_pretrained('bert-base-uncased')
                self.bert_model.eval()
            except Exception as e:
                print(f"Warning: Could not load BERT model: {e}")
                self.use_bert = False
        
        # Initialize spaCy
        if use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                print(f"Warning: Could not load spaCy model: {e}")
                self.use_spacy = False
        
        # Initialize TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        # Musical keywords
        self.musical_keywords = {
            'emotion': ['happy', 'sad', 'melancholic', 'energetic', 'calm', 'intense', 'peaceful', 'dramatic'],
            'genre': ['pop', 'rock', 'jazz', 'classical', 'blues', 'country', 'electronic', 'folk'],
            'instruments': ['piano', 'guitar', 'drums', 'bass', 'violin', 'saxophone', 'trumpet', 'flute'],
            'tempo': ['fast', 'slow', 'moderate', 'lively', 'relaxed', 'upbeat', 'downbeat'],
            'dynamics': ['loud', 'quiet', 'soft', 'strong', 'gentle', 'powerful', 'delicate']
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text
    
    def extract_musical_features(self, text: str) -> Dict[str, Any]:
        """Extract musical features from text."""
        text_lower = text.lower()
        features = {}
        
        # Extract emotions
        features['emotions'] = [word for word in self.musical_keywords['emotion'] 
                              if word in text_lower]
        
        # Extract genres
        features['genres'] = [word for word in self.musical_keywords['genre'] 
                            if word in text_lower]
        
        # Extract instruments
        features['instruments'] = [word for word in self.musical_keywords['instruments'] 
                                 if word in text_lower]
        
        # Extract tempo indicators
        features['tempo'] = [word for word in self.musical_keywords['tempo'] 
                           if word in text_lower]
        
        # Extract dynamics
        features['dynamics'] = [word for word in self.musical_keywords['dynamics'] 
                              if word in text_lower]
        
        return features
    
    def get_bert_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get BERT embedding for text."""
        if not self.use_bert:
            return None
        
        try:
            # Tokenize
            inputs = self.bert_tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=True
            )
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].numpy()
            
            return embedding.flatten()
        except Exception as e:
            print(f"Error getting BERT embedding: {e}")
            return None
    
    def get_spacy_features(self, text: str) -> Dict[str, Any]:
        """Get spaCy NLP features."""
        if not self.use_spacy:
            return {}
        
        try:
            doc = self.nlp(text)
            
            features = {
                'entities': [(ent.text, ent.label_) for ent in doc.ents],
                'noun_chunks': [chunk.text for chunk in doc.noun_chunks],
                'pos_tags': [token.pos_ for token in doc],
                'sentiment': doc.sentiment if hasattr(doc, 'sentiment') else None
            }
            
            return features
        except Exception as e:
            print(f"Error getting spaCy features: {e}")
            return {}
    
    def get_tfidf_features(self, text: str) -> np.ndarray:
        """Get TF-IDF features for text."""
        try:
            # Fit and transform
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([text])
            return tfidf_matrix.toarray().flatten()
        except Exception as e:
            print(f"Error getting TF-IDF features: {e}")
            return np.zeros(1000)  # Default size
    
    def process_text(self, text: str) -> Dict[str, Any]:
        """Process a single text description."""
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Extract features
        musical_features = self.extract_musical_features(cleaned_text)
        spacy_features = self.get_spacy_features(cleaned_text)
        bert_embedding = self.get_bert_embedding(cleaned_text)
        tfidf_features = self.get_tfidf_features(cleaned_text)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'musical_features': musical_features,
            'spacy_features': spacy_features,
            'bert_embedding': bert_embedding.tolist() if bert_embedding is not None else None,
            'tfidf_features': tfidf_features.tolist(),
            'text_length': len(cleaned_text),
            'word_count': len(cleaned_text.split())
        }
    
    def process_text_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Process a batch of text descriptions."""
        processed_texts = []
        
        for text in texts:
            processed = self.process_text(text)
            processed_texts.append(processed)
        
        return processed_texts
    
    def create_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts."""
        embeddings = []
        
        for text in texts:
            embedding = self.get_bert_embedding(text)
            if embedding is not None:
                embeddings.append(embedding)
            else:
                # Fallback to TF-IDF
                tfidf = self.get_tfidf_features(text)
                embeddings.append(tfidf)
        
        return np.array(embeddings)
    
    def cluster_texts(self, texts: List[str], n_clusters: int = 5) -> List[int]:
        """Cluster texts based on their embeddings."""
        embeddings = self.create_text_embeddings(texts)
        
        if len(embeddings) < n_clusters:
            return list(range(len(embeddings)))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)
        
        return clusters.tolist()
    
    def save_processed_texts(self, processed_texts: List[Dict[str, Any]], output_file: str):
        """Save processed text data to file."""
        import os
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_texts, f, indent=2, ensure_ascii=False)
        
        print(f"Processed text data saved to {output_file}")
        print(f"Total processed texts: {len(processed_texts)}") 