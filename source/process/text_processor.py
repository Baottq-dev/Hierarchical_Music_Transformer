"""
Text Processor - Processes text descriptions for training
"""

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import spacy
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import BertModel, BertTokenizer


class TextProcessor:
    """Processes text descriptions for training."""

    def __init__(
        self,
        max_length: int = 512,
        use_bert: bool = True,
        use_spacy: bool = True,
        use_gpu: bool = True,
        use_cache: bool = True,
        cache_dir: str = "data/processed/text_cache",
        batch_size: int = 32,
    ):
        self.max_length = max_length
        self.use_bert = use_bert
        self.use_spacy = use_spacy
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.batch_size = batch_size

        # Create cache directory if needed
        if use_cache and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        # Check for GPU availability
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        print(f"TextProcessor: Using device: {self.device}")

        # Initialize BERT
        if use_bert:
            try:
                self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                self.bert_model = BertModel.from_pretrained("bert-base-uncased")
                self.bert_model.to(self.device)  # Move model to GPU if available
                self.bert_model.eval()
                print(f"BERT model loaded on {self.device}")
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
            max_features=1000, stop_words="english", ngram_range=(1, 2)
        )

        # Musical keywords
        self.musical_keywords = {
            "emotion": [
                "happy",
                "sad",
                "melancholic",
                "energetic",
                "calm",
                "intense",
                "peaceful",
                "dramatic",
            ],
            "genre": ["pop", "rock", "jazz", "classical", "blues", "country", "electronic", "folk"],
            "instruments": [
                "piano",
                "guitar",
                "drums",
                "bass",
                "violin",
                "saxophone",
                "trumpet",
                "flute",
            ],
            "tempo": ["fast", "slow", "moderate", "lively", "relaxed", "upbeat", "downbeat"],
            "dynamics": ["loud", "quiet", "soft", "strong", "gentle", "powerful", "delicate"],
        }

    def _get_cache_path(self, text: str) -> str:
        """Get cache file path for a text."""
        if not self.use_cache:
            return None

        # Create a hash of the text to avoid issues with special characters
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{text_hash}.json")

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s.,!?-]", "", text)

        # Convert to lowercase
        text = text.lower()

        return text

    def extract_musical_features(self, text: str) -> Dict[str, Any]:
        """Extract musical features from text."""
        text_lower = text.lower()
        features = {}

        # Extract emotions
        features["emotions"] = [
            word for word in self.musical_keywords["emotion"] if word in text_lower
        ]

        # Extract genres
        features["genres"] = [word for word in self.musical_keywords["genre"] if word in text_lower]

        # Extract instruments
        features["instruments"] = [
            word for word in self.musical_keywords["instruments"] if word in text_lower
        ]

        # Extract tempo indicators
        features["tempo"] = [word for word in self.musical_keywords["tempo"] if word in text_lower]

        # Extract dynamics
        features["dynamics"] = [
            word for word in self.musical_keywords["dynamics"] if word in text_lower
        ]

        return features

    def get_bert_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get BERT embedding for text."""
        if not self.use_bert:
            return None

        try:
            # Tokenize
            inputs = self.bert_tokenizer(
                text, return_tensors="pt", max_length=self.max_length, truncation=True, padding=True
            )

            # Move inputs to device (GPU if available)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token embedding
                embedding = (
                    outputs.last_hidden_state[:, 0, :].cpu().numpy()
                )  # Move back to CPU for numpy conversion

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
                "entities": [(ent.text, ent.label_) for ent in doc.ents],
                "noun_chunks": [chunk.text for chunk in doc.noun_chunks],
                "pos_tags": [token.pos_ for token in doc],
                "sentiment": doc.sentiment if hasattr(doc, "sentiment") else None,
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
        # Check cache first if enabled
        cache_path = self._get_cache_path(text)
        if self.use_cache and cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache for text: {e}")

        # Clean text
        cleaned_text = self.clean_text(text)

        # Extract features
        musical_features = self.extract_musical_features(cleaned_text)
        spacy_features = self.get_spacy_features(cleaned_text)
        bert_embedding = self.get_bert_embedding(cleaned_text)
        tfidf_features = self.get_tfidf_features(cleaned_text)

        # Create result
        result = {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "musical_features": musical_features,
            "spacy_features": spacy_features,
            "bert_embedding": bert_embedding.tolist() if bert_embedding is not None else None,
            "tfidf_features": tfidf_features.tolist(),
            "text_length": len(cleaned_text),
            "word_count": len(cleaned_text.split()),
        }

        # Save to cache if enabled
        if self.use_cache and cache_path:
            try:
                with open(cache_path, "w") as f:
                    json.dump(result, f)
            except Exception as e:
                print(f"Error saving cache for text: {e}")

        return result

    def process_text_batch(
        self, texts: List[str], show_progress: bool = True
    ) -> List[Dict[str, Any]]:
        """Process a batch of text descriptions efficiently."""
        processed_texts = []

        # Check which texts are already cached
        uncached_texts = []
        uncached_indices = []
        cached_results = [None] * len(texts)

        if self.use_cache:
            for i, text in enumerate(texts):
                cache_path = self._get_cache_path(text)
                if cache_path and os.path.exists(cache_path):
                    try:
                        with open(cache_path) as f:
                            cached_results[i] = json.load(f)
                    except Exception:
                        uncached_texts.append(text)
                        uncached_indices.append(i)
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))

        # Process uncached texts
        if uncached_texts:
            # Clean all texts first
            cleaned_texts = [self.clean_text(text) for text in uncached_texts]

            # Extract musical features
            musical_features = [self.extract_musical_features(text) for text in cleaned_texts]

            # Get BERT embeddings in batches
            bert_embeddings = self.batch_bert_embeddings(cleaned_texts)

            # Get spaCy features (can be slow)
            spacy_features = []
            if show_progress and self.use_spacy:
                for text in tqdm(cleaned_texts, desc="SpaCy processing"):
                    spacy_features.append(self.get_spacy_features(text))
            else:
                spacy_features = [self.get_spacy_features(text) for text in cleaned_texts]

            # Get TF-IDF features
            tfidf_features = []
            for text in cleaned_texts:
                tfidf_features.append(self.get_tfidf_features(text).tolist())

            # Combine all features
            for i, (text, cleaned, musical, spacy, bert, tfidf) in enumerate(
                zip(
                    uncached_texts,
                    cleaned_texts,
                    musical_features,
                    spacy_features,
                    bert_embeddings,
                    tfidf_features,
                )
            ):
                result = {
                    "original_text": text,
                    "cleaned_text": cleaned,
                    "musical_features": musical,
                    "spacy_features": spacy,
                    "bert_embedding": bert.tolist() if bert is not None else None,
                    "tfidf_features": tfidf,
                    "text_length": len(cleaned),
                    "word_count": len(cleaned.split()),
                }

                # Save to cache
                if self.use_cache:
                    cache_path = self._get_cache_path(text)
                    if cache_path:
                        try:
                            with open(cache_path, "w") as f:
                                json.dump(result, f)
                        except Exception as e:
                            print(f"Error saving cache: {e}")

                # Store at the correct position
                cached_results[uncached_indices[i]] = result

        # Return all results in original order
        return cached_results

    def batch_bert_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Get BERT embeddings for a batch of texts efficiently."""
        if not self.use_bert:
            return [None] * len(texts)

        embeddings = []

        # Process in batches to avoid memory issues
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i : i + self.batch_size]

            try:
                # Tokenize batch
                inputs = self.bert_tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    max_length=self.max_length,
                    truncation=True,
                    padding=True,
                )

                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get embeddings
                with torch.no_grad():
                    outputs = self.bert_model(**inputs)
                    # Use [CLS] token embedding for each text
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

                # Add to results
                embeddings.extend([emb for emb in batch_embeddings])

            except Exception as e:
                print(f"Error batch processing BERT embeddings: {e}")
                # Add None for each text in the failed batch
                embeddings.extend([None] * len(batch_texts))

        return embeddings

    def process_texts_parallel(
        self,
        texts: List[str],
        batch_size: int = 100,
        checkpoint_interval: int = 10,
        checkpoint_file: str = "data/processed/text_checkpoint.json",
        show_progress: bool = True,
    ) -> List[Dict[str, Any]]:
        """Process multiple texts with batching and checkpointing."""

        # Check for existing checkpoint
        processed_texts = []
        last_processed_idx = -1

        if os.path.exists(checkpoint_file):
            try:
                with open(checkpoint_file) as f:
                    checkpoint = json.load(f)
                    processed_texts = checkpoint.get("processed_texts", [])
                    last_processed_idx = checkpoint.get("last_processed_idx", -1)
                    print(
                        f"Resuming from checkpoint: {last_processed_idx + 1}/{len(texts)} texts processed"
                    )
            except Exception as e:
                print(f"Error loading checkpoint: {e}")

        # If already completed, return results
        if last_processed_idx >= len(texts) - 1:
            print("All texts already processed.")
            return processed_texts

        # Split into batches
        remaining_texts = texts[last_processed_idx + 1 :]
        batches = [
            remaining_texts[i : i + batch_size] for i in range(0, len(remaining_texts), batch_size)
        ]

        print(f"Processing {len(remaining_texts)} texts in {len(batches)} batches")

        # Process each batch
        for batch_idx, batch_texts in enumerate(batches):
            batch_start_time = time.time()
            print(f"Processing batch {batch_idx + 1}/{len(batches)}...")

            # Process batch
            batch_results = self.process_text_batch(batch_texts, show_progress)

            # Add results to processed texts
            processed_texts.extend(batch_results)
            current_idx = last_processed_idx + len(processed_texts)

            # Calculate performance metrics
            batch_time = time.time() - batch_start_time
            texts_per_second = len(batch_results) / batch_time if batch_time > 0 else 0

            print(
                f"Batch {batch_idx + 1} completed in {batch_time:.2f}s ({texts_per_second:.2f} texts/s)"
            )

            # Save checkpoint
            checkpoint = {
                "processed_texts": processed_texts,
                "last_processed_idx": current_idx,
                "total_texts": len(texts),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "batch_info": {
                    "current_batch": batch_idx + 1,
                    "total_batches": len(batches),
                    "batch_size": batch_size,
                    "processing_time": batch_time,
                    "texts_per_second": texts_per_second,
                },
            }

            # Ensure directory exists
            os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

            # Save checkpoint
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint, f)

            print(f"Checkpoint saved: {current_idx + 1}/{len(texts)} texts processed")

            # Stop after checkpoint_interval batches if specified
            if checkpoint_interval > 0 and (batch_idx + 1) >= checkpoint_interval:
                print(f"Reached {checkpoint_interval} batches, stopping as requested")
                break

            # Estimate remaining time
            if batch_idx < len(batches) - 1:
                remaining_batches = len(batches) - batch_idx - 1
                est_time_remaining = remaining_batches * batch_time
                print(
                    f"Estimated time remaining: {est_time_remaining/3600:.1f} hours ({est_time_remaining/60:.1f} minutes)"
                )

        return processed_texts

    def create_text_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts."""
        if self.use_bert:
            # Use batched processing for efficiency
            bert_embeddings = self.batch_bert_embeddings(texts)
            embeddings = []

            for embedding in bert_embeddings:
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    # Fallback to TF-IDF
                    embeddings.append(np.zeros(768))

            return np.array(embeddings)
        else:
            # Use TF-IDF as fallback
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            return tfidf_matrix.toarray()

    def cluster_texts(self, texts: List[str], n_clusters: int = 5) -> List[int]:
        """Cluster texts based on their embeddings."""
        # Create embeddings
        embeddings = self.create_text_embeddings(texts)

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embeddings)

        return clusters.tolist()

    def save_processed_texts(self, processed_texts: List[Dict[str, Any]], output_file: str):
        """Save processed texts to file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w") as f:
            json.dump(processed_texts, f)

        print(f"Saved {len(processed_texts)} processed texts to {output_file}")
