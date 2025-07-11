"""
Integration test for the data processing pipeline.
"""

import os
import pytest
import json
import numpy as np

from amt.process import MIDIProcessor, TextProcessor, DataPreparer


def json_serializable(obj):
    """Convert objects to JSON serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class TestProcessingPipeline:
    """Test suite for the processing pipeline."""
    
    def test_midi_to_tokens_pipeline(self, simple_midi_file, temp_dir):
        """Test the pipeline from MIDI file to tokens."""
        # Initialize processor
        processor = MIDIProcessor(use_cache=False)
        
        # Process MIDI file
        result = processor.process_midi_file(simple_midi_file)
        
        # Check result
        assert result is not None
        assert 'tokens' in result
        assert len(result['tokens']) > 0
        
        # Save processed data
        output_file = os.path.join(temp_dir, "processed_midi.json")
        processor.save_processed_data([result], output_file)
        
        # Check output file
        assert os.path.exists(output_file)
        
        # Load the saved data
        with open(output_file, 'r') as f:
            loaded_data = json.load(f)
        
        # Check loaded data
        assert len(loaded_data) == 1
        assert 'tokens' in loaded_data[0]
        assert loaded_data[0]['tokens'] == result['tokens']
    
    def test_simple_processing_pipeline(self, simple_midi_file, temp_dir):
        """Test a simple processing pipeline with MIDI and text."""
        # Create a simple text file
        text_file = os.path.join(temp_dir, "test.txt")
        with open(text_file, 'w') as f:
            f.write("This is a test description for a simple piano melody.")
        
        # Initialize processors
        midi_processor = MIDIProcessor(use_cache=False)
        text_processor = TextProcessor(use_cache=False)
        
        # Process MIDI file
        midi_result = midi_processor.process_midi_file(simple_midi_file)
        
        # Read text file and process text content
        with open(text_file, 'r') as f:
            text_content = f.read()
        text_result = text_processor.process_text(text_content)
        
        # Print the text result for debugging
        print("Text result keys:", list(text_result.keys()))
        
        # Check results
        assert midi_result is not None
        assert text_result is not None
        
        assert 'tokens' in midi_result
        assert 'cleaned_text' in text_result
        
        # Create a data pair
        data_pair = {
            'pair_id': 'test_pair',
            'midi_data': midi_result,
            'text_data': text_result,
            'metadata': {
                'midi_file': simple_midi_file,
                'text_file': text_file
            }
        }
        
        # Save the pair
        pair_file = os.path.join(temp_dir, "pair.json")
        with open(pair_file, 'w') as f:
            json.dump(data_pair, f, default=json_serializable)
        
        # Initialize data preparer with correct parameters
        data_preparer = DataPreparer(
            max_sequence_length=1024,
            max_text_length=512,
            batch_size=32
        )
        
        # Create training data
        train_file = os.path.join(temp_dir, "train.json")
        val_file = os.path.join(temp_dir, "val.json")
        test_file = os.path.join(temp_dir, "test.json")
        
        # Prepare a mock dataset with our pair
        dataset = [data_pair]
        
        # Use the prepare_training_data method directly with the dataset
        # Since we don't have a paired_data_file, we'll need to modify the test
        # to use the split_data method directly
        train_data, val_data, test_data = data_preparer.split_data(dataset)
        
        # Save the split data
        with open(train_file, 'w') as f:
            json.dump(train_data, f, default=json_serializable)
        with open(val_file, 'w') as f:
            json.dump(val_data, f, default=json_serializable)
        with open(test_file, 'w') as f:
            json.dump(test_data, f, default=json_serializable)
        
        # Check output files
        assert os.path.exists(train_file)
        
        # Load and check training data
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        
        # Verify the structure
        assert isinstance(train_data, list)
        # Note: Since we only have one item, it might be in train, val, or test
        # So we can't assert on the length of train_data 