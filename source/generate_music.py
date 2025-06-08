import json
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import mido
import os
import numpy as np

# --- Configuration ---
MODEL_DIR = "./data/output/amt_model_fine_tuned"
GENERATION_OUTPUT_DIR = "./data/output/generated_music"
MAX_GENERATION_LENGTH = 256 # Number of new events to generate after the prompt
TICKS_PER_BEAT = 480
DEFAULT_TEMPO = 500000 # Microseconds per beat (120 BPM)

# Constants from preprocess_midi.py for event conversion
TIME_RESOLUTION = 0.01  # 10ms steps for time shifts
# MAX_TIME_SHIFT_STEPS = int(1.0 / TIME_RESOLUTION) -1 # Max 99 steps (not directly used here, but good to remember)
VELOCITY_BINS = 32
MAX_VELOCITY = 127
VELOCITY_BIN_SIZE = (MAX_VELOCITY + 1) / VELOCITY_BINS # Should be 128 / 32 = 4.0


# --- 1. Load Model and Tokenizer ---
def load_model_and_tokenizer(model_dir):
    print(f"Loading tokenizer from {model_dir}...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    print(f"Loading model from {model_dir}...")
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval() # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model and tokenizer loaded on {device}.")
    return model, tokenizer, device

# --- 2. Event Sequence to MIDI ---
def event_sequence_to_midi(event_sequence, output_midi_path):
    mid = mido.MidiFile(ticks_per_beat=TICKS_PER_BEAT)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(mido.MetaMessage("set_tempo", tempo=DEFAULT_TEMPO, time=0))

    current_velocity = 64 # Default MIDI velocity (0-127)
    accumulated_delay_ticks = 0

    for event_str in event_sequence:
        parts = event_str.split("_")
        event_main_type = parts[0]

        if event_str in tokenizer.all_special_tokens:
            # print(f"Ignoring special token: {event_str}")
            continue

        try:
            if event_main_type == "SET" and parts[1] == "VELOCITY":
                v_bin = int(parts[-1])
                current_velocity = int(v_bin * VELOCITY_BIN_SIZE + VELOCITY_BIN_SIZE / 2.0) # Midpoint of bin
                current_velocity = max(0, min(MAX_VELOCITY, current_velocity)) # Clamp to 0-127
            elif event_main_type == "TIME" and parts[1] == "SHIFT":
                dt_steps = int(parts[-1])
                delay_seconds = dt_steps * TIME_RESOLUTION
                delay_ticks = int(mido.second2tick(delay_seconds, TICKS_PER_BEAT, DEFAULT_TEMPO))
                accumulated_delay_ticks += delay_ticks
            elif event_main_type == "NOTE" and parts[1] == "ON":
                pitch = int(parts[-1])
                msg = mido.Message("note_on", note=pitch, velocity=current_velocity, time=accumulated_delay_ticks)
                track.append(msg)
                accumulated_delay_ticks = 0
            elif event_main_type == "NOTE" and parts[1] == "OFF":
                pitch = int(parts[-1])
                msg = mido.Message("note_off", note=pitch, velocity=0, time=accumulated_delay_ticks)
                track.append(msg)
                accumulated_delay_ticks = 0
            elif event_main_type == "SEMANTIC" and parts[1] == "TOKEN":
                pass # These are for conditioning, not part of MIDI output itself
            else:
                print(f"Warning: Unknown or unhandled event string format: {event_str}")
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not parse event 	{event_str}	: {e}")
            continue

    mid.save(output_midi_path)
    print(f"MIDI file saved to {output_midi_path}")

# --- 3. Generate Music ---
def generate_music_for_semantic_token(model, tokenizer, device, semantic_token_str, num_sequences=1):
    generated_event_sequences = []
    
    try:
        semantic_token_id = tokenizer.convert_tokens_to_ids(semantic_token_str)
    except KeyError:
        print(f"Error: Semantic token 	{semantic_token_str}	 not in tokenizer vocabulary.")
        return []

    prompt_ids = [tokenizer.bos_token_id, semantic_token_id]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long).to(device)

    print(f"Generating with prompt: {tokenizer.decode(prompt_ids)}")

    with torch.no_grad(): # Disable gradient calculations for inference
        output_sequences_ids = model.generate(
            input_ids=input_ids,
            max_length=MAX_GENERATION_LENGTH + len(prompt_ids),
            num_return_sequences=num_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.9,
            early_stopping=True
        )

    for i, generated_ids_tensor in enumerate(output_sequences_ids):
        generated_ids = generated_ids_tensor.tolist()
        
        # Decode the full sequence
        full_decoded_tokens = tokenizer.convert_ids_to_tokens(generated_ids)
        
        # Remove the prompt part to get only the newly generated events
        # Prompt was [BOS, SEMANTIC_TOKEN_X]
        # Find the end of the prompt in the decoded tokens
        # The prompt tokens are tokenizer.bos_token and semantic_token_str
        
        # Simpler: generated IDs start with input_ids. Slice after that.
        start_index_of_generation = input_ids.shape[-1]
        purely_generated_ids = generated_ids[start_index_of_generation:]
        
        # Decode only the purely generated part, and stop at EOS/PAD
        actual_generated_events = []
        for token_id in purely_generated_ids:
            if token_id in [tokenizer.eos_token_id, tokenizer.pad_token_id]:
                break
            actual_generated_events.append(tokenizer.convert_ids_to_tokens(token_id))

        generated_event_sequences.append(actual_generated_events)
        print(f"Generated sequence {i+1} for {semantic_token_str} (first 20 events): {actual_generated_events[:20]}")

    return generated_event_sequences

# --- Main ---
if __name__ == "__main__":
    os.makedirs(GENERATION_OUTPUT_DIR, exist_ok=True)
    
    global tokenizer # Make tokenizer global for event_sequence_to_midi to access special tokens
    model, tokenizer, device = load_model_and_tokenizer(MODEL_DIR)

    if model is None or tokenizer is None:
        print("Exiting due to model/tokenizer loading failure.")
        exit()
    
    if tokenizer.pad_token is None:
        print("Warning: PAD token is not set in the tokenizer. Adding [PAD].")
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        # No need to resave tokenizer here as it is only for generation script's use

    # Semantic tokens to test (K=3 was used in clustering for the small sample)
    semantic_tokens_to_try = []
    for i in range(3):
        token_str = f"SEMANTIC_TOKEN_{i}"
        try:
            # Check if token exists by trying to convert (will raise KeyError if not)
            tokenizer.convert_tokens_to_ids(token_str)
            semantic_tokens_to_try.append(token_str)
        except KeyError:
            print(f"Warning: Test token {token_str} not found in tokenizer vocabulary. Skipping.")
            
    if not semantic_tokens_to_try:
        print("No valid SEMANTIC_TOKEN_X found in vocab. Cannot run generation as intended.")
        exit()

    for sem_token in semantic_tokens_to_try:
        print(f"\n--- Generating music for: {sem_token} ---")
        event_sequences = generate_music_for_semantic_token(model, tokenizer, device, sem_token, num_sequences=1)
        
        for i, seq in enumerate(event_sequences):
            if seq:
                # Sanitize filename from token string
                safe_sem_token_name = sem_token.replace("_", "")
                output_filename = f"generated_for_{safe_sem_token_name}_seq{i+1}.mid"
                output_path = os.path.join(GENERATION_OUTPUT_DIR, output_filename)
                event_sequence_to_midi(seq, output_path)
            else:
                print(f"No sequence generated for {sem_token}, sequence index {i}")
    
    print(f"\nAll generation attempts complete. MIDI files (if any) are in {GENERATION_OUTPUT_DIR}")
    print("Please note: Due to the very small dataset and limited training, the quality of generated music might be low.")

