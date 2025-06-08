import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import os

# --- Configuration ---
MODEL_OUTPUT_DIR = "./data/output/amt_model_fine_tuned"
TRAINING_DATA_PATH = "./data/output/amt_training_data.json"
VOCAB_FILE = os.path.join(MODEL_OUTPUT_DIR, "vocab.json")
TOKENIZER_CONFIG_FILE = os.path.join(MODEL_OUTPUT_DIR, "tokenizer_config.json") # For custom tokenizer info

# Model Hyperparameters (kept small for demonstration)
NUM_EPOCHS = 3 # Very few epochs due to small dataset and sandbox limits
BATCH_SIZE = 1 # Smallest possible batch size
LEARNING_RATE = 5e-5
MAX_SEQ_LENGTH = 512 # Max sequence length for GPT-2

# --- 1. Vocabulary and Tokenizer Preparation ---
def create_or_load_tokenizer(training_data_path, vocab_file, tokenizer_config_file, model_output_dir):
    if os.path.exists(vocab_file) and os.path.exists(os.path.join(model_output_dir, "tokenizer.json")):
        print(f"Loading existing tokenizer from {model_output_dir}")
        tokenizer = GPT2Tokenizer.from_pretrained(model_output_dir)
        # Ensure PAD token is set if it was added
        if tokenizer.pad_token is None:
            print("Setting PAD token for loaded tokenizer.")
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        return tokenizer

    print("Creating new tokenizer...")
    try:
        with open(training_data_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading training data for vocab: {e}")
        return None

    all_tokens = set()
    for item in data:
        all_tokens.update(item["combined_sequence_for_amt"])
    
    # Create vocab mapping
    # Add special tokens: PAD for padding, UNK for unknown (though GPT2 uses BPE)
    # GPT2Tokenizer handles BPE, but we are creating a vocab from scratch for symbolic data.
    # For simplicity with symbolic data, we can treat each event as a whole token.
    # However, GPT2Tokenizer expects a vocab.json (char/subword to id) and merges.txt for BPE.
    # A simpler approach for fully symbolic vocab: build a map and save it.
    # For this demo, we will create a vocab.json and use it to initialize GPT2Tokenizer.
    # This is a bit of a hack for GPT2Tokenizer which is designed for BPE on text.

    # Let's ensure PAD token is part of the vocab from the start
    special_tokens = {"pad_token": "[PAD]", "eos_token": "[EOS]", "bos_token": "[BOS]", "unk_token": "[UNK]"}
    sorted_tokens = sorted(list(all_tokens))
    
    # Create vocab dictionary (token to ID)
    vocab_map = {token: i for i, token in enumerate(sorted_tokens)}
    # Add special tokens, ensuring they don't overwrite existing IDs if names clash by chance
    # Start special token IDs after existing tokens
    current_id = len(vocab_map)
    for special_name, special_token_str in special_tokens.items():
        if special_token_str not in vocab_map:
            vocab_map[special_token_str] = current_id
            current_id +=1

    os.makedirs(model_output_dir, exist_ok=True)
    with open(vocab_file, "w") as f:
        json.dump(vocab_map, f, indent=4)
    
    # GPT2Tokenizer needs merges.txt, even if empty for a non-BPE vocab. Create a dummy one.
    with open(os.path.join(model_output_dir, "merges.txt"), "w") as f:
        pass # Empty merges file

    # Save a tokenizer_config.json to specify special tokens
    tokenizer_conf = {
        "model_max_length": MAX_SEQ_LENGTH,
        **{k:v for k,v in special_tokens.items()} # Add pad_token, eos_token etc.
    }
    with open(os.path.join(model_output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_conf, f, indent=4)

    # Initialize tokenizer from our custom vocab and config
    try:
        tokenizer = GPT2Tokenizer(vocab_file=vocab_file, 
                                  merges_file=os.path.join(model_output_dir, "merges.txt"),
                                  tokenizer_file=None, # Avoids looking for a full tokenizer.json from Hugging Face
                                  **special_tokens # Pass special tokens directly
                                 )
        # Save the tokenizer so it can be loaded with from_pretrained
        tokenizer.save_pretrained(model_output_dir)
        print(f"Tokenizer created and saved to {model_output_dir}")
        print(f"Vocab size: {tokenizer.vocab_size}")
        if tokenizer.pad_token is None:
             print("PAD token is still None after save_pretrained. Manually setting.")
             tokenizer.add_special_tokens({"pad_token": "[PAD]"})
             tokenizer.save_pretrained(model_output_dir) # Save again

    except Exception as e:
        print(f"Error initializing or saving tokenizer: {e}")
        return None
    
    return tokenizer

# --- 2. Dataset Class ---
class AMTDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.sequences = []

        for item in data:
            # The combined sequence already includes the semantic token
            token_sequence = item["combined_sequence_for_amt"]
            
            # Convert symbolic tokens to IDs. 
            # For GPT2, input_ids and labels are typically the same for language modeling.
            # Add BOS and EOS tokens to frame the sequence for the model
            # Using tokenizer.bos_token and tokenizer.eos_token strings
            # The tokenizer should convert these to their respective IDs.
            
            # We need to encode the list of string tokens. 
            # GPT2Tokenizer.encode() takes a string, not a list of pre-tokenized strings.
            # So, we join them into a single string, then tokenize. This is standard.
            # However, our tokens are symbolic and might contain spaces or special chars
            # that BPE would break down. 
            # A better way for pre-tokenized symbolic data is to convert directly to IDs.
            
            input_ids = [tokenizer.bos_token_id] + [tokenizer.convert_tokens_to_ids(tok) for tok in token_sequence] + [tokenizer.eos_token_id]
            
            # Truncate if longer than max_length
            input_ids = input_ids[:self.max_length]
            
            self.sequences.append(input_ids)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)

# --- 3. Collate Function for DataLoader (handles padding) ---
def collate_fn(batch, pad_token_id):
    batch_tensors = [item for item in batch]
    # Pad sequences to the max length in the batch
    batch_tensors_padded = torch.nn.utils.rnn.pad_sequence(batch_tensors, batch_first=True, padding_value=pad_token_id)
    
    # For GPT-2 language modeling, labels are typically the input_ids shifted by one position.
    # However, Hugging Face GPT2LMHeadModel handles this internally if labels are not provided,
    # or if labels=input_ids. We will provide labels = input_ids.
    return {"input_ids": batch_tensors_padded, "labels": batch_tensors_padded}

# --- 4. Main Fine-tuning Logic ---
def fine_tune_amt():
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    # Create or load tokenizer
    tokenizer = create_or_load_tokenizer(TRAINING_DATA_PATH, VOCAB_FILE, TOKENIZER_CONFIG_FILE, MODEL_OUTPUT_DIR)
    if tokenizer is None:
        print("Failed to create or load tokenizer. Exiting.")
        return
    
    if tokenizer.pad_token_id is None:
        print("Error: PAD token ID is None. Padding will fail.")
        # This should have been handled by create_or_load_tokenizer
        # Attempt to fix again if somehow missed
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
        if tokenizer.pad_token_id is None:
             print("CRITICAL: PAD token ID still None. Cannot proceed.")
             return
        print(f"PAD token ID successfully set to: {tokenizer.pad_token_id}")

    # Load training data
    try:
        with open(TRAINING_DATA_PATH, "r") as f:
            raw_training_data = json.load(f)
    except Exception as e:
        print(f"Error loading AMT training data: {e}")
        return

    if not raw_training_data:
        print("No training data found. Exiting.")
        return

    # For demonstration, we use all data for training as it is very small
    train_data = raw_training_data 
    # val_data, train_data = train_test_split(raw_training_data, test_size=0.1, random_state=42) # if we had more data

    train_dataset = AMTDataset(train_data, tokenizer, max_length=MAX_SEQ_LENGTH)
    # val_dataset = AMTDataset(val_data, tokenizer, max_length=MAX_SEQ_LENGTH) # if we had validation

    if len(train_dataset) == 0:
        print("Training dataset is empty after processing. Exiting.")
        return

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                                collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))
    # val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id))

    # Model Configuration
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=MAX_SEQ_LENGTH,
        n_ctx=MAX_SEQ_LENGTH,
        n_embd=256, # Small embedding size for demo
        n_layer=4,  # Small number of layers
        n_head=4,   # Small number of heads
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    model = GPT2LMHeadModel(config=config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Starting fine-tuning for {NUM_EPOCHS} epochs on {device}...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            if loss is None:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}: Loss is None. Skipping batch.")
                continue
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_dataloader)}, Loss: {loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_epoch_loss:.4f}")

    # Save the fine-tuned model and tokenizer
    model.save_pretrained(MODEL_OUTPUT_DIR)
    # Tokenizer already saved during creation, but can save again if any changes were made (e.g. pad token)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR)
    print(f"Fine-tuned model and tokenizer saved to {MODEL_OUTPUT_DIR}")

if __name__ == "__main__":
    fine_tune_amt()

