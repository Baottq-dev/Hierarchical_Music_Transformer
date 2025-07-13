"""
Model Evaluator - Evaluates model performance and generated music
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Tuple

import pretty_midi

from amt.evaluate.metrics import EvaluationMetrics
from amt.process.midi_processor import MidiProcessor


class ModelEvaluator:
    """Evaluates the quality of generated music."""

    def __init__(self):
        self.metrics = {}

    def evaluate_single_file(self, midi_file: str) -> Dict[str, Any]:
        """Evaluate a single MIDI file."""
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_file)
            return self._calculate_metrics(midi_data)
        except Exception as e:
            print(f"Error evaluating {midi_file}: {e}")
            return {}

    def _calculate_metrics(self, midi_data: pretty_midi.PrettyMIDI) -> Dict[str, Any]:
        """Calculate various musical metrics."""
        metrics = {}

        # Basic statistics
        metrics["duration"] = midi_data.get_end_time()
        metrics["tempo"] = midi_data.estimate_tempo()
        metrics["instrument_count"] = len(midi_data.instruments)

        # Note statistics
        all_notes = []
        for instrument in midi_data.instruments:
            all_notes.extend(instrument.notes)

        if all_notes:
            pitches = [note.pitch for note in all_notes]
            velocities = [note.velocity for note in all_notes]
            durations = [note.end - note.start for note in all_notes]

            metrics["note_count"] = len(all_notes)
            metrics["pitch_range"] = max(pitches) - min(pitches)
            metrics["avg_pitch"] = np.mean(pitches)
            metrics["pitch_std"] = np.std(pitches)
            metrics["avg_velocity"] = np.mean(velocities)
            metrics["velocity_std"] = np.std(velocities)
            metrics["avg_duration"] = np.mean(durations)
            metrics["duration_std"] = np.std(durations)

            # Note density
            metrics["note_density"] = len(all_notes) / metrics["duration"]

            # Pitch distribution
            metrics["pitch_distribution"] = np.histogram(pitches, bins=12, range=(0, 127))[
                0
            ].tolist()

            # Velocity distribution
            metrics["velocity_distribution"] = np.histogram(velocities, bins=16, range=(0, 128))[
                0
            ].tolist()

        # Time signature
        if midi_data.time_signature_changes:
            ts = midi_data.time_signature_changes[0]
            metrics["time_signature"] = f"{ts.numerator}/{ts.denominator}"
        else:
            metrics["time_signature"] = "4/4"

        # Key signature
        if midi_data.key_signature_changes:
            ks = midi_data.key_signature_changes[0]
            metrics["key_signature"] = ks.key_number
        else:
            metrics["key_signature"] = 0

        return metrics

    def evaluate_generated_vs_reference(
        self, generated_file: str, reference_file: str
    ) -> Dict[str, Any]:
        """Compare generated music with reference music."""
        gen_metrics = self.evaluate_single_file(generated_file)
        ref_metrics = self.evaluate_single_file(reference_file)

        if not gen_metrics or not ref_metrics:
            return {}

        # Calculate similarity scores
        similarities = {}

        # Pitch similarity
        if "pitch_distribution" in gen_metrics and "pitch_distribution" in ref_metrics:
            gen_pitch = np.array(gen_metrics["pitch_distribution"])
            ref_pitch = np.array(ref_metrics["pitch_distribution"])
            similarities["pitch_similarity"] = cosine_similarity(
                gen_pitch.reshape(1, -1), ref_pitch.reshape(1, -1)
            )[0][0]

        # Velocity similarity
        if "velocity_distribution" in gen_metrics and "velocity_distribution" in ref_metrics:
            gen_vel = np.array(gen_metrics["velocity_distribution"])
            ref_vel = np.array(ref_metrics["velocity_distribution"])
            similarities["velocity_similarity"] = cosine_similarity(
                gen_vel.reshape(1, -1), ref_vel.reshape(1, -1)
            )[0][0]

        # Duration similarity
        gen_dur = gen_metrics.get("duration", 0)
        ref_dur = ref_metrics.get("duration", 0)
        if gen_dur > 0 and ref_dur > 0:
            similarities["duration_similarity"] = 1 - abs(gen_dur - ref_dur) / max(gen_dur, ref_dur)

        # Tempo similarity
        gen_tempo = gen_metrics.get("tempo", 120)
        ref_tempo = ref_metrics.get("tempo", 120)
        similarities["tempo_similarity"] = 1 - abs(gen_tempo - ref_tempo) / max(
            gen_tempo, ref_tempo
        )

        # Note density similarity
        gen_density = gen_metrics.get("note_density", 0)
        ref_density = ref_metrics.get("note_density", 0)
        if gen_density > 0 and ref_density > 0:
            similarities["density_similarity"] = 1 - abs(gen_density - ref_density) / max(
                gen_density, ref_density
            )

        return {
            "generated_metrics": gen_metrics,
            "reference_metrics": ref_metrics,
            "similarities": similarities,
            "overall_similarity": np.mean(list(similarities.values())) if similarities else 0,
        }

    def evaluate_batch(
        self, generated_files: List[str], reference_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Evaluate a batch of generated files."""
        results = {"individual_metrics": [], "batch_statistics": {}, "comparison_results": []}

        # Evaluate individual files
        for file in generated_files:
            metrics = self.evaluate_single_file(file)
            if metrics:
                results["individual_metrics"].append({"file": file, "metrics": metrics})

        # Calculate batch statistics
        if results["individual_metrics"]:
            all_metrics = [item["metrics"] for item in results["individual_metrics"]]

            # Aggregate statistics
            batch_stats = {}
            for key in all_metrics[0].keys():
                if isinstance(all_metrics[0][key], (int, float)):
                    values = [m.get(key, 0) for m in all_metrics]
                    batch_stats[f"{key}_mean"] = np.mean(values)
                    batch_stats[f"{key}_std"] = np.std(values)
                    batch_stats[f"{key}_min"] = np.min(values)
                    batch_stats[f"{key}_max"] = np.max(values)

            results["batch_statistics"] = batch_stats

        # Compare with reference files if provided
        if reference_files and len(reference_files) == len(generated_files):
            for gen_file, ref_file in zip(generated_files, reference_files):
                comparison = self.evaluate_generated_vs_reference(gen_file, ref_file)
                if comparison:
                    results["comparison_results"].append(
                        {
                            "generated_file": gen_file,
                            "reference_file": ref_file,
                            "comparison": comparison,
                        }
                    )

        return results

    def generate_evaluation_report(
        self, evaluation_results: Dict[str, Any], output_file: str = "evaluation_report.json"
    ):
        """Generate a comprehensive evaluation report."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

        print(f"Evaluation report saved to {output_file}")

        # Print summary
        if evaluation_results["individual_metrics"]:
            print("\nEvaluation Summary:")
            print(f"  Files evaluated: {len(evaluation_results['individual_metrics'])}")

            if evaluation_results["batch_statistics"]:
                stats = evaluation_results["batch_statistics"]
                print(f"  Average duration: {stats.get('duration_mean', 0):.2f}s")
                print(f"  Average note count: {stats.get('note_count_mean', 0):.1f}")
                print(f"  Average pitch range: {stats.get('pitch_range_mean', 0):.1f}")

        if evaluation_results["comparison_results"]:
            similarities = [
                r["comparison"]["overall_similarity"]
                for r in evaluation_results["comparison_results"]
            ]
            print(f"  Average similarity to reference: {np.mean(similarities):.3f}")

    def plot_metrics(
        self, evaluation_results: Dict[str, Any], output_dir: str = "evaluation_plots"
    ):
        """Generate plots for evaluation metrics."""
        os.makedirs(output_dir, exist_ok=True)

        if not evaluation_results["individual_metrics"]:
            return

        # Extract metrics
        metrics_list = [item["metrics"] for item in evaluation_results["individual_metrics"]]

        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Music Generation Evaluation Metrics", fontsize=16)

        # Duration distribution
        durations = [m.get("duration", 0) for m in metrics_list]
        axes[0, 0].hist(durations, bins=20, alpha=0.7)
        axes[0, 0].set_title("Duration Distribution")
        axes[0, 0].set_xlabel("Duration (seconds)")
        axes[0, 0].set_ylabel("Count")

        # Note count distribution
        note_counts = [m.get("note_count", 0) for m in metrics_list]
        axes[0, 1].hist(note_counts, bins=20, alpha=0.7)
        axes[0, 1].set_title("Note Count Distribution")
        axes[0, 1].set_xlabel("Note Count")
        axes[0, 1].set_ylabel("Count")

        # Pitch range distribution
        pitch_ranges = [m.get("pitch_range", 0) for m in metrics_list]
        axes[0, 2].hist(pitch_ranges, bins=20, alpha=0.7)
        axes[0, 2].set_title("Pitch Range Distribution")
        axes[0, 2].set_xlabel("Pitch Range")
        axes[0, 2].set_ylabel("Count")

        # Average velocity distribution
        avg_velocities = [m.get("avg_velocity", 0) for m in metrics_list]
        axes[1, 0].hist(avg_velocities, bins=20, alpha=0.7)
        axes[1, 0].set_title("Average Velocity Distribution")
        axes[1, 0].set_xlabel("Average Velocity")
        axes[1, 0].set_ylabel("Count")

        # Note density distribution
        note_densities = [m.get("note_density", 0) for m in metrics_list]
        axes[1, 1].hist(note_densities, bins=20, alpha=0.7)
        axes[1, 1].set_title("Note Density Distribution")
        axes[1, 1].set_xlabel("Notes per Second")
        axes[1, 1].set_ylabel("Count")

        # Tempo distribution
        tempos = [m.get("tempo", 120) for m in metrics_list]
        axes[1, 2].hist(tempos, bins=20, alpha=0.7)
        axes[1, 2].set_title("Tempo Distribution")
        axes[1, 2].set_xlabel("Tempo (BPM)")
        axes[1, 2].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, "metrics_distribution.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"Metrics plots saved to {output_dir}")


def evaluate_batch(
    generated_files: List[str],
    reference_files: Optional[List[str]] = None,
    output_dir: str = "evaluation",
) -> Dict[str, Any]:
    """Convenience function to evaluate a batch of files."""
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_batch(generated_files, reference_files)

    # Generate report
    evaluator.generate_evaluation_report(
        results, os.path.join(output_dir, "evaluation_report.json")
    )

    # Generate plots
    evaluator.plot_metrics(results, output_dir)

    return results
