from __future__ import annotations

"""
Model Tester - Utility to load checkpoints, run simple sanity checks and benchmark generation speed.
Designed to satisfy run_test.py expectations (test_model_loading, run_comprehensive_test, benchmark_performance).
NOTE: This is *not* a full test harness – it provides lightweight functionality so that
run_test.py can execute without ImportError.
"""

import os
import time
from typing import Any, Dict, List

import torch

from ..process.midi_processor import MIDIProcessor
from ..process.text_processor import TextProcessor
from ..train.model import MusicTransformer  # relative import


class ModelTester:
    """Light-weight tester for MusicTransformer checkpoints."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: MusicTransformer | None = None

    # --------------------------------------------------
    #  Public API expected by run_test.py
    # --------------------------------------------------
    def test_model_loading(self) -> Dict[str, Any]:
        """Try to load a checkpoint; return dict with success + model_info."""
        if not os.path.exists(self.checkpoint_path):
            return {"success": False, "error": "Checkpoint not found"}

        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

            # Retrieve model config – fallback to defaults if missing
            model_cfg = checkpoint.get("model_config", {})
            self.model = MusicTransformer(
                vocab_size=model_cfg.get("vocab_size", 125),
                d_model=model_cfg.get("d_model", 512),
                n_heads=model_cfg.get("n_heads", 8),
                n_layers=model_cfg.get("n_layers", 6),
                d_ff=model_cfg.get("d_ff", 2048),
                max_seq_len=model_cfg.get("max_seq_len", 1024),
                max_text_len=model_cfg.get("max_text_len", 512),
                use_cross_attention=True,
            )
            self.model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            self.model.to(self.device).eval()

            return {
                "success": True,
                "model_info": self.model.get_model_info()
                if hasattr(self.model, "get_model_info")
                else None,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_comprehensive_test(self, output_dir: str) -> Dict[str, Any]:
        """Placeholder: simply checks model forward pass on dummy data."""
        results: Dict[str, Any] = {"pipeline_integration": {"success": False, "errors": []}}
        if self.model is None:
            res = self.test_model_loading()
            if not res.get("success"):
                results["pipeline_integration"]["errors"].append(res.get("error", "unknown"))
                return results

        try:
            # Dummy sample – 1 token BOS + pads
            midi_dummy = torch.zeros(1, 32, dtype=torch.long, device=self.device)
            text_dummy = torch.zeros(1, 512, 768, device=self.device)
            _ = self.model(midi_dummy, text_dummy)
            results["pipeline_integration"]["success"] = True
        except Exception as e:
            results["pipeline_integration"]["errors"].append(str(e))

        return results

    def benchmark_performance(self, num_samples: int = 10) -> Dict[str, Any]:
        """Generate *num_samples* short sequences and report average time/length."""
        if self.model is None:
            res = self.test_model_loading()
            if not res.get("success"):
                return {"success": False, "error": "model not loaded"}

        self.model.eval()
        tp = TextProcessor(max_length=128, use_gpu=self.device.type == "cuda", use_cache=False)
        midi_proc = MIDIProcessor(max_sequence_length=128)
        times: List[float] = []
        lengths: List[int] = []

        with torch.no_grad():
            for _ in range(num_samples):
                text_emb = tp.get_bert_embedding("a calm piano intro")
                if text_emb is None:
                    text_tensor = torch.zeros(1, 128, 768, device=self.device)
                else:
                    t = torch.tensor(text_emb, dtype=torch.float32, device=self.device).unsqueeze(0)
                    text_tensor = t.unsqueeze(1).repeat(1, 128, 1)

                start = time.time()
                generated = self.model.generate(text_tensor, max_length=128)
                times.append(time.time() - start)
                lengths.append(generated.shape[1])

        return {
            "success": True,
            "avg_generation_time": float(sum(times) / len(times)) if times else 0,
            "avg_sequence_length": float(sum(lengths) / len(lengths)) if lengths else 0,
        }
