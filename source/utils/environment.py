import torch
import transformers
import sklearn
import mido
import pretty_midi
import pandas

def verify_environment():
    """Verifies and prints versions of required packages."""
    print(f"PyTorch version: {torch.__version__}")
    print(f"Transformers version: {transformers.__version__}")
    print(f"scikit-learn version: {sklearn.__version__}")
    print(f"Mido version: {mido.__version__}")
    print(f"pretty_midi version: {pretty_midi.__version__}")
    print(f"Pandas version: {pandas.__version__}")

if __name__ == "__main__":
    verify_environment() 