import json
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import re

def clean_text(text):
    text = re.sub(r'\[n\]', ' ', text)  # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text)    # Replace multiple spaces with single space
    text = text.strip()
    return text

def get_bert_embeddings(text_list, model_name='bert-base-uncased'):
    """
    Generates BERT embeddings for a list of texts.
    Returns a list of embeddings (numpy arrays).
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    model.eval()  # Put model in evaluation mode

    embeddings_list = []
    with torch.no_grad(): # Disable gradient calculations for inference
        for text in text_list:
            cleaned_text = clean_text(text)
            # Tokenize input text and add special tokens ([CLS] and [SEP])
            inputs = tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True, max_length=512)
            # Get hidden states from the model
            outputs = model(**inputs)
            # Use the embedding of the [CLS] token as the sentence embedding
            # Last hidden state shape: (batch_size, sequence_length, hidden_size)
            cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
            embeddings_list.append(cls_embedding)
    return embeddings_list

if __name__ == "__main__":
    input_json_path = "./data/output/automated_paired_data.json"
    output_json_path = "./data/output/text_embeddings.json"

    try:
        with open(input_json_path, 'r') as f:
            paired_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file {input_json_path} not found.")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {input_json_path}.")
        exit()

    if not paired_data:
        print("No data found in the input JSON file.")
        exit()

    text_descriptions = [item.get("text_description", "") for item in paired_data]
    file_paths = [item.get("file_path", "") for item in paired_data]
    artists = [item.get("artist", "") for item in paired_data]
    titles = [item.get("title", "") for item in paired_data]

    print(f"Found {len(text_descriptions)} text descriptions to process.")

    # Generate embeddings
    # For demonstration, let's process only the first few if the list is long
    # In a real scenario, you'd process all or batch them.
    # For now, we process all available in the sample file.
    embeddings = get_bert_embeddings(text_descriptions)

    print(f"Generated {len(embeddings)} embeddings.")

    # Store embeddings with corresponding file paths
    output_data = []
    for i in range(len(paired_data)):
        output_data.append({
            "file_path": file_paths[i],
            "artist": artists[i],
            "title": titles[i],
            "text_description": text_descriptions[i],
            "embedding": embeddings[i].tolist() # Convert numpy array to list for JSON serialization
        })

    try:
        with open(output_json_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Successfully saved text embeddings to {output_json_path}")
    except IOError:
        print(f"Error: Could not write to output file {output_json_path}.")


