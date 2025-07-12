"""
Text Processor - Processes text descriptions for training and generation
"""

import hashlib
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Optional imports
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Some text processing features will be disabled.")

try:
    from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer, RobertaModel, RobertaTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. BERT embeddings will be disabled.")

try:
    import sentencepiece as spm
    SENTENCEPIECE_AVAILABLE = True
except ImportError:
    SENTENCEPIECE_AVAILABLE = False
    print("Warning: sentencepiece not available. SentencePiece tokenization will be disabled.")

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. TF-IDF and clustering will be disabled.")

from amt.utils.logging import get_logger

logger = get_logger(__name__)


class TextProcessor:
    """Processes text descriptions for training."""

    def __init__(
        self,
        max_length: int = 512,
        use_bert: bool = True,
        use_spacy: bool = True,
        use_sentencepiece: bool = True,
        bert_model_name: str = "bert-base-uncased",
        sentencepiece_model: str = None,
        use_gpu: bool = True,
        use_cache: bool = True,
        cache_dir: str = "data/processed/text_cache",
        batch_size: int = 32,
        enable_fine_tuning: bool = False,
        music_fine_tuned_model: str = None,
        use_pretrained_model: bool = False,
        pretrained_model_path: Optional[str] = None,
        optimal_transfer_learning: bool = False
    ):
        self.max_length = max_length
        self.use_bert = use_bert and TRANSFORMERS_AVAILABLE
        self.use_spacy = use_spacy and SPACY_AVAILABLE
        self.use_sentencepiece = use_sentencepiece and SENTENCEPIECE_AVAILABLE
        self.bert_model_name = bert_model_name
        self.sentencepiece_model = sentencepiece_model
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.enable_fine_tuning = enable_fine_tuning
        self.music_fine_tuned_model = music_fine_tuned_model
        self.use_pretrained_model = use_pretrained_model
        self.pretrained_model_path = pretrained_model_path
        self.optimal_transfer_learning = optimal_transfer_learning

        # Create cache directory if needed
        if use_cache and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        # Check for GPU availability
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        logger.info(f"TextProcessor: Using device: {self.device}")

        # Initialize BERT or pretrained model
        if self.use_pretrained_model and self.pretrained_model_path:
            self._load_pretrained_model()
        elif self.use_bert:
            self._initialize_bert()

        # Initialize SentencePiece
        if self.use_sentencepiece:
            try:
                if self.sentencepiece_model:
                    self.sp_processor = spm.SentencePieceProcessor()
                    self.sp_processor.load(self.sentencepiece_model)
                    logger.info(f"SentencePiece model loaded from {self.sentencepiece_model}")
                else:
                    # If no model is provided, we'll need to train one or use default
                    self.sp_processor = None
                    logger.warning("No SentencePiece model provided. Will train on first use or use BERT tokenizer.")
            except Exception as e:
                logger.error(f"Warning: Could not load SentencePiece model: {e}")
                self.use_sentencepiece = False

        # Initialize spaCy
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception as e:
                logger.error(f"Warning: Could not load spaCy model: {e}")
                self.use_spacy = False

        # Initialize TF-IDF
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000, stop_words="english", ngram_range=(1, 2)
            )
        else:
            self.tfidf_vectorizer = None

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
    
    def _initialize_bert(self):
        """Initialize BERT model"""
        try:
            # Use RoBERTa if optimal_transfer_learning is enabled
            if self.optimal_transfer_learning:
                logger.info("Using RoBERTa-base model for optimal transfer learning")
                self.bert_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
                self.bert_model = RobertaModel.from_pretrained("roberta-base")
                self.bert_model_name = "roberta-base"
            # Use fine-tuned music model if specified and available
            elif self.enable_fine_tuning and self.music_fine_tuned_model and os.path.exists(self.music_fine_tuned_model):
                logger.info(f"Loading fine-tuned music language model from {self.music_fine_tuned_model}")
                self.bert_tokenizer = AutoTokenizer.from_pretrained(self.music_fine_tuned_model)
                self.bert_model = AutoModel.from_pretrained(self.music_fine_tuned_model)
            # Use specified model name
            elif self.pretrained_model_path:
                logger.info(f"Loading specified pretrained model: {self.pretrained_model_path}")
                if 'roberta' in self.pretrained_model_path.lower():
                    self.bert_tokenizer = RobertaTokenizer.from_pretrained(self.pretrained_model_path)
                    self.bert_model = RobertaModel.from_pretrained(self.pretrained_model_path)
                elif 'bert' in self.pretrained_model_path.lower():
                    self.bert_tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
                    self.bert_model = BertModel.from_pretrained(self.pretrained_model_path)
                else:
                    # For other Hugging Face models
                    self.bert_tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_path)
                    self.bert_model = AutoModel.from_pretrained(self.pretrained_model_path)
            # Use default BERT model
            elif 'bert' in self.bert_model_name.lower():
                self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
                self.bert_model = BertModel.from_pretrained(self.bert_model_name)
            else:
                # For other Hugging Face models
                self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
                self.bert_model = AutoModel.from_pretrained(self.bert_model_name)
            
            self.bert_model.to(self.device)  # Move model to GPU if available
            
            # Set model to eval mode if not fine-tuning
            if not self.enable_fine_tuning:
                self.bert_model.eval()
                # Freeze parameters
                for param in self.bert_model.parameters():
                    param.requires_grad = False
            
            logger.info(f"Transformer model {self.bert_model_name} loaded on {self.device}")
        except Exception as e:
            logger.error(f"Warning: Could not load Transformer model: {e}")
            self.use_bert = False
    
    def _load_pretrained_model(self):
        """Load pretrained text model for feature extraction"""
        try:
            logger.info(f"Loading pretrained text model from {self.pretrained_model_path}")
            
            # If optimal_transfer_learning is enabled, use RoBERTa-base
            if self.optimal_transfer_learning:
                logger.info("Using RoBERTa-base model for optimal transfer learning")
                self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
                self.pretrained_model = RobertaModel.from_pretrained("roberta-base")
                self.pretrained_model.to(self.device)
                return
            
            # Check if it's a Hugging Face model name or local path
            if os.path.exists(self.pretrained_model_path):
                # Local path - load using torch.load
                self.pretrained_model = torch.load(self.pretrained_model_path, map_location=self.device)
                # Try to load tokenizer if available
                tokenizer_path = os.path.join(os.path.dirname(self.pretrained_model_path), "tokenizer")
                if os.path.exists(tokenizer_path):
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
                else:
                    # Fallback to BERT tokenizer
                    self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            else:
                # Hugging Face model name
                if 'roberta' in self.pretrained_model_path.lower():
                    self.tokenizer = RobertaTokenizer.from_pretrained(self.pretrained_model_path)
                    self.pretrained_model = RobertaModel.from_pretrained(self.pretrained_model_path)
                elif 'bert' in self.pretrained_model_path.lower():
                    self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_model_path)
                    self.pretrained_model = BertModel.from_pretrained(self.pretrained_model_path)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_path)
                    self.pretrained_model = AutoModel.from_pretrained(self.pretrained_model_path)
                self.pretrained_model.to(self.device)
            
            # Set to eval mode if not fine-tuning
            if not self.enable_fine_tuning:
                self.pretrained_model.eval()
                # Freeze parameters
                for param in self.pretrained_model.parameters():
                    param.requires_grad = False
            
            logger.info(f"Successfully loaded pretrained text model")
        except Exception as e:
            logger.error(f"Error loading pretrained model: {e}")
            self.pretrained_model = None
            # Fallback to BERT if available
            if TRANSFORMERS_AVAILABLE:
                logger.info("Falling back to default BERT model")
                self._initialize_bert()

    def extract_features_with_pretrained(self, text):
        """Extract features using pretrained model"""
        if not hasattr(self, 'pretrained_model') or self.pretrained_model is None:
            return self.extract_features_default(text)
        
        try:
            # Clean text
            text = self.clean_text(text)
            
            # Tokenize text
            if hasattr(self, 'tokenizer'):
                inputs = self.tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
            else:
                # Fallback to BERT tokenizer
                inputs = self.bert_tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            with torch.no_grad():
                outputs = self.pretrained_model(**inputs)
                
            # Get embeddings
            token_embeddings = outputs.last_hidden_state.cpu().squeeze().numpy()
            
            # Get sentence embedding (CLS token or pooled output)
            if hasattr(outputs, 'pooler_output'):
                sentence_embedding = outputs.pooler_output.cpu().squeeze().numpy()
            else:
                # Use CLS token as sentence embedding
                sentence_embedding = token_embeddings[0]
            
            return {
                "token_embeddings": token_embeddings,
                "sentence_embedding": sentence_embedding,
                "input_ids": inputs["input_ids"].cpu().squeeze().tolist()
            }
        except Exception as e:
            logger.error(f"Error extracting features with pretrained model: {e}")
            return self.extract_features_default(text)

    def extract_features_default(self, text):
        """Default feature extraction when no pretrained model is available"""
        # Clean text
        text = self.clean_text(text)
        
        # Tokenize with basic tokenizer
        tokens = text.split()
        
        # Create simple embeddings
        embedding_dim = 768
        vocab_size = 30000
        
        # Use consistent random seed for reproducibility
        np.random.seed(42)
        embedding_matrix = np.random.randn(vocab_size, embedding_dim) * 0.02
        
        # Map tokens to IDs using hash function
        token_ids = [hash(token) % vocab_size for token in tokens]
        
        # Create token embeddings
        if len(token_ids) > 0:
            token_embeddings = np.array([embedding_matrix[token_id] for token_id in token_ids])
            # Pad or truncate to max_length
            if len(token_embeddings) > self.max_length:
                token_embeddings = token_embeddings[:self.max_length]
            elif len(token_embeddings) < self.max_length:
                padding = np.zeros((self.max_length - len(token_embeddings), embedding_dim))
                token_embeddings = np.vstack([token_embeddings, padding])
            
            # Get sentence embedding by averaging token embeddings
            sentence_embedding = np.mean(token_embeddings, axis=0)
        else:
            # Handle empty text
            token_embeddings = np.zeros((self.max_length, embedding_dim))
            sentence_embedding = np.zeros(embedding_dim)
            token_ids = [0] * min(len(tokens), self.max_length)
        
        return {
            "token_embeddings": token_embeddings,
            "sentence_embedding": sentence_embedding,
            "input_ids": token_ids
        }
    
    def get_vocab_size(self):
        """Get vocabulary size of the tokenizer"""
        if hasattr(self, 'tokenizer'):
            return len(self.tokenizer)
        elif hasattr(self, 'bert_tokenizer'):
            return len(self.bert_tokenizer)
        else:
            return 30000  # Default size
        
    def process(self, text):
        """Process text into tokens and features"""
        # Extract features based on available models
        if self.use_pretrained_model and hasattr(self, 'pretrained_model') and self.pretrained_model is not None:
            features = self.extract_features_with_pretrained(text)
        elif self.use_bert and hasattr(self, 'bert_model') and self.bert_model is not None:
            features = self.get_bert_embedding(text)
        else:
            features = self.extract_features_default(text)
        
        # Extract musical features
        musical_features = self.extract_musical_features(text)
        
        # Combine all features
        result = {
            "text": text,
            "tokens": features.get("input_ids", []),
            "features": features,
            "musical_features": musical_features
        }
        
        return result

    def fine_tune_language_model(
        self, 
        texts: List[str], 
        output_model_path: str = "models/checkpoints/music_bert",
        num_epochs: int = 3, 
        batch_size: int = 8,
        learning_rate: float = 2e-5
    ):
        """
        Fine-tune the language model on music descriptions
        
        Args:
            texts: List of music description texts
            output_model_path: Path to save the fine-tuned model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for fine-tuning
        
        Returns:
            True if fine-tuning was successful
        """
        if not TRANSFORMERS_AVAILABLE:
            logger.error("Transformers library not available. Cannot fine-tune model.")
            return False
            
        try:
            from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
            
            logger.info(f"Fine-tuning language model on {len(texts)} music descriptions")
            
            # Create directory for output model
            os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
            
            # Prepare tokenizer and model for fine-tuning
            tokenizer = self.bert_tokenizer
            model = self.bert_model
            
            # Set model to training mode
            model.train()
            
            # Tokenize the texts
            def tokenize_function(examples):
                return tokenizer(examples, padding="max_length", truncation=True, max_length=self.max_length)
            
            # Create dataset
            from torch.utils.data import Dataset
            
            class MusicTextDataset(Dataset):
                def __init__(self, texts, tokenizer):
                    self.encodings = tokenize_function(texts)
                    
                def __getitem__(self, idx):
                    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                
                def __len__(self):
                    return len(self.encodings.input_ids)
            
            # Split into train/val
            from sklearn.model_selection import train_test_split
            train_texts, val_texts = train_test_split(texts, test_size=0.1)
            
            train_dataset = MusicTextDataset(train_texts, tokenizer)
            val_dataset = MusicTextDataset(val_texts, tokenizer)
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer, mlm=True, mlm_probability=0.15
            )
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_model_path,
                overwrite_output_dir=True,
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                learning_rate=learning_rate,
                weight_decay=0.01,
                logging_dir=os.path.join(output_model_path, "logs"),
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=data_collator,
            )
            
            # Train the model
            trainer.train()
            
            # Save the fine-tuned model
            model.save_pretrained(output_model_path)
            tokenizer.save_pretrained(output_model_path)
            
            # Update the instance to use the fine-tuned model
            self.music_fine_tuned_model = output_model_path
            self.bert_model = model
            self.bert_tokenizer = tokenizer
            
            logger.info(f"Fine-tuned model saved to {output_model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error fine-tuning language model: {e}")
            return False

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

    def train_sentencepiece_model(self, texts: List[str], model_prefix: str = "sp_model", vocab_size: int = 8000):
        """Train a SentencePiece model on the provided texts."""
        if not SENTENCEPIECE_AVAILABLE:
            logger.error("SentencePiece not available. Cannot train model.")
            return False
        
        try:
            # Create temporary file with all texts
            temp_file = os.path.join(self.cache_dir, "sp_training_data.txt")
            with open(temp_file, 'w', encoding='utf-8') as f:
                for text in texts:
                    f.write(text + "\n")
            
            # Train model
            model_path = os.path.join(self.cache_dir, model_prefix)
            spm.SentencePieceTrainer.train(
                input=temp_file,
                model_prefix=model_path,
                vocab_size=vocab_size,
                model_type='unigram',  # or 'bpe', 'char', 'word'
                character_coverage=0.9995,
                normalization_rule_name='nmt_nfkc_cf'
            )
            
            # Load the trained model
            self.sentencepiece_model = f"{model_path}.model"
            self.sp_processor = spm.SentencePieceProcessor()
            self.sp_processor.load(self.sentencepiece_model)
            self.use_sentencepiece = True
            
            logger.info(f"SentencePiece model trained and saved to {self.sentencepiece_model}")
            return True
        except Exception as e:
            logger.error(f"Error training SentencePiece model: {e}")
            return False

    def get_sentencepiece_tokens(self, text: str) -> List[str]:
        """Tokenize text using SentencePiece."""
        if not self.use_sentencepiece or not self.sp_processor:
            return []
        
        try:
            tokens = self.sp_processor.encode_as_pieces(text)
            return tokens
        except Exception as e:
            logger.error(f"Error tokenizing with SentencePiece: {e}")
            return []

    def get_bert_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get BERT embedding for text."""
        if not self.use_bert or not TRANSFORMERS_AVAILABLE:
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
            logger.error(f"Error getting BERT embedding: {e}")
            return None

    def get_hybrid_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get hybrid embedding using SentencePiece tokens + BERT."""
        if not (self.use_bert and self.use_sentencepiece):
            # Fall back to regular BERT if both aren't available
            return self.get_bert_embedding(text)
        
        try:
            # Get SentencePiece tokens
            if self.sp_processor is None:
                # Initialize SentencePiece if not already done
                logger.info("Training SentencePiece model on first use")
                if not self.train_sentencepiece_model([text], vocab_size=4000):
                    # Fall back to BERT if SentencePiece training fails
                    return self.get_bert_embedding(text)
            
            sp_tokens = self.get_sentencepiece_tokens(text)
            if not sp_tokens:
                return self.get_bert_embedding(text)
                
            # Reconstruct text from SentencePiece tokens
            # This helps maintain sentence boundaries while benefiting from SentencePiece segmentation
            tokenized_text = " ".join(sp_tokens)
            
            # Process with BERT
            inputs = self.bert_tokenizer(
                tokenized_text, 
                return_tensors="pt", 
                max_length=self.max_length, 
                truncation=True, 
                padding=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                
                # Get full contextual embeddings for all tokens
                token_embeddings = outputs.last_hidden_state[0].cpu().numpy()
                
                # Use mean pooling for all tokens (excluding special tokens)
                # This captures the contextual meaning better than just the CLS token
                attention_mask = inputs['attention_mask'].cpu().numpy()[0]
                # Get mean of all token embeddings, weighted by attention mask
                mean_embedding = np.sum(token_embeddings * attention_mask[:, np.newaxis], axis=0) / np.sum(attention_mask)
                
            return mean_embedding
        except Exception as e:
            logger.error(f"Error getting hybrid embedding: {e}")
            # Fall back to regular BERT
            return self.get_bert_embedding(text)

    def get_spacy_features(self, text: str) -> Dict[str, Any]:
        """Get spaCy features for text."""
        if not self.use_spacy or not SPACY_AVAILABLE:
            return {}

        try:
            doc = self.nlp(text)
            features = {
                "entities": [ent.text for ent in doc.ents],
                "pos_tags": {token.text: token.pos_ for token in doc},
                "noun_chunks": [chunk.text for chunk in doc.noun_chunks],
            }
            return features
        except Exception as e:
            logger.error(f"Error getting spaCy features: {e}")
            return {}

    def get_tfidf_features(self, text: str) -> np.ndarray:
        """Get TF-IDF features for text."""
        if not SKLEARN_AVAILABLE or not self.tfidf_vectorizer:
            return np.array([])

        try:
            # Fit on text (not optimal but will work for single text)
            tfidf = self.tfidf_vectorizer.fit_transform([text])
            return tfidf.toarray()[0]
        except Exception as e:
            logger.error(f"Error getting TF-IDF features: {e}")
            return np.array([])

    def process_text(self, text: str) -> Dict[str, Any]:
        """Process a single text."""
        # Check cache first
        cache_path = self._get_cache_path(text)
        if self.use_cache and cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading from cache: {e}")

        # Clean text
        cleaned_text = self.clean_text(text)

        # Initialize result dict
        result = {
            "original_text": text,
            "cleaned_text": cleaned_text,
        }

        # Extract musical features
        result["musical_features"] = self.extract_musical_features(cleaned_text)

        # Get SentencePiece tokens if available
        if self.use_sentencepiece and SENTENCEPIECE_AVAILABLE:
            result["sentencepiece_tokens"] = self.get_sentencepiece_tokens(cleaned_text)

        # Get hybrid embedding (SentencePiece + BERT)
        embedding = self.get_hybrid_embedding(cleaned_text)
        if embedding is not None:
            result["embedding"] = embedding.tolist()

        # Get spaCy features
        spacy_features = self.get_spacy_features(cleaned_text)
        if spacy_features:
            result["spacy_features"] = spacy_features

        # Save to cache
        if self.use_cache and cache_path:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    json.dump(result, f)
            except Exception as e:
                logger.warning(f"Error saving to cache: {e}")

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
                for text in uncached_texts: # Use uncached_texts directly
                    spacy_features.append(self.get_spacy_features(text))
            else:
                spacy_features = [self.get_spacy_features(text) for text in uncached_texts]

            # Get TF-IDF features
            tfidf_features = []
            for text in uncached_texts:
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
            if SKLEARN_AVAILABLE and self.tfidf_vectorizer is not None:
                tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                return tfidf_matrix.toarray()
            else:
                return np.zeros((len(texts), 1000)) # Default size

    def cluster_texts(self, texts: List[str], n_clusters: int = 5) -> List[int]:
        """Cluster texts based on their embeddings."""
        # Create embeddings
        embeddings = self.create_text_embeddings(texts)

        if not SKLEARN_AVAILABLE or self.tfidf_vectorizer is None:
            print("Warning: TF-IDF or scikit-learn not available, cannot perform clustering.")
            return [0] * len(texts) # Return dummy clusters

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
