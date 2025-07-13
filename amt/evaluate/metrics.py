"""
Evaluation Metrics - Musical metrics for evaluating generated music
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import pretty_midi

from amt.process.midi_processor import MidiProcessor


class EvaluationMetrics:
    """Comprehensive music evaluation metrics."""

    def __init__(self):
        self.metric_names = [
            "note_density",
            "velocity_diversity",
            "pitch_range",
            "rhythm_consistency",
            "melodic_contour",
            "harmonic_complexity",
            "tempo_stability",
            "dynamic_range",
            "structural_coherence",
        ]

    def calculate_note_density(self, midi_data: pretty_midi.PrettyMIDI) -> float:
        """Calculate note density (notes per second)."""
        all_notes = []
        for instrument in midi_data.instruments:
            all_notes.extend(instrument.notes)

        duration = midi_data.get_end_time()
        if duration > 0:
            return len(all_notes) / duration
        return 0.0

    def calculate_velocity_diversity(self, midi_data: pretty_midi.PrettyMIDI) -> float:
        """Calculate velocity diversity (standard deviation of velocities)."""
        all_velocities = []
        for instrument in midi_data.instruments:
            all_velocities.extend([note.velocity for note in instrument.notes])

        if all_velocities:
            return np.std(all_velocities)
        return 0.0

    def calculate_pitch_range(self, midi_data: pretty_midi.PrettyMIDI) -> Tuple[int, int]:
        """Calculate pitch range (min and max pitches)."""
        all_pitches = []
        for instrument in midi_data.instruments:
            all_pitches.extend([note.pitch for note in instrument.notes])

        if all_pitches:
            return min(all_pitches), max(all_pitches)
        return 60, 60  # Default to middle C

    def calculate_rhythm_consistency(self, midi_data: pretty_midi.PrettyMIDI) -> float:
        """Calculate rhythm consistency (standard deviation of note durations)."""
        all_durations = []
        for instrument in midi_data.instruments:
            all_durations.extend([note.end - note.start for note in instrument.notes])

        if all_durations:
            return np.std(all_durations)
        return 0.0

    def calculate_melodic_contour(self, midi_data: pretty_midi.PrettyMIDI) -> float:
        """Calculate melodic contour complexity."""
        all_notes = []
        for instrument in midi_data.instruments:
            all_notes.extend(instrument.notes)

        if len(all_notes) < 2:
            return 0.0

        # Sort notes by start time
        all_notes.sort(key=lambda x: x.start)

        # Calculate pitch differences between consecutive notes
        pitch_diffs = []
        for i in range(1, len(all_notes)):
            diff = all_notes[i].pitch - all_notes[i - 1].pitch
            pitch_diffs.append(abs(diff))

        if pitch_diffs:
            return np.mean(pitch_diffs)
        return 0.0

    def calculate_harmonic_complexity(self, midi_data: pretty_midi.PrettyMIDI) -> float:
        """Calculate harmonic complexity (number of simultaneous notes)."""
        # Get all note events
        events = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                events.append((note.start, "start", note.pitch))
                events.append((note.end, "end", note.pitch))

        # Sort events by time
        events.sort(key=lambda x: x[0])

        # Count simultaneous notes
        active_notes = set()
        max_simultaneous = 0

        for time, event_type, pitch in events:
            if event_type == "start":
                active_notes.add(pitch)
                max_simultaneous = max(max_simultaneous, len(active_notes))
            else:
                active_notes.discard(pitch)

        return max_simultaneous

    def calculate_tempo_stability(self, midi_data: pretty_midi.PrettyMIDI) -> float:
        """Calculate tempo stability."""
        if midi_data.tempo_changes:
            tempos = [change.tempo for change in midi_data.tempo_changes]
            if len(tempos) > 1:
                return np.std(tempos)

        return 0.0  # Stable tempo

    def calculate_dynamic_range(self, midi_data: pretty_midi.PrettyMIDI) -> float:
        """Calculate dynamic range (max velocity - min velocity)."""
        all_velocities = []
        for instrument in midi_data.instruments:
            all_velocities.extend([note.velocity for note in instrument.notes])

        if all_velocities:
            return max(all_velocities) - min(all_velocities)
        return 0.0

    def calculate_structural_coherence(self, midi_data: pretty_midi.PrettyMIDI) -> float:
        """Calculate structural coherence (repetition patterns)."""
        # Extract note sequences
        all_notes = []
        for instrument in midi_data.instruments:
            all_notes.extend(instrument.notes)

        if len(all_notes) < 4:
            return 0.0

        # Sort by time
        all_notes.sort(key=lambda x: x.start)

        # Create pitch sequence
        pitch_sequence = [note.pitch for note in all_notes]

        # Calculate autocorrelation to find repetition patterns
        autocorr = np.correlate(pitch_sequence, pitch_sequence, mode="full")
        autocorr = autocorr[len(pitch_sequence) - 1 :]

        # Normalize
        autocorr = autocorr / autocorr[0]

        # Find peaks in autocorrelation (indicating repetition)
        peaks = []
        for i in range(1, len(autocorr) - 1):
            if autocorr[i] > autocorr[i - 1] and autocorr[i] > autocorr[i + 1]:
                peaks.append(autocorr[i])

        if peaks:
            return np.mean(peaks)
        return 0.0

    def calculate_all_metrics(self, midi_data: pretty_midi.PrettyMIDI) -> Dict[str, float]:
        """Calculate all metrics for a MIDI file."""
        metrics = {}

        metrics["note_density"] = self.calculate_note_density(midi_data)
        metrics["velocity_diversity"] = self.calculate_velocity_diversity(midi_data)

        pitch_min, pitch_max = self.calculate_pitch_range(midi_data)
        metrics["pitch_range"] = pitch_max - pitch_min
        metrics["pitch_min"] = pitch_min
        metrics["pitch_max"] = pitch_max

        metrics["rhythm_consistency"] = self.calculate_rhythm_consistency(midi_data)
        metrics["melodic_contour"] = self.calculate_melodic_contour(midi_data)
        metrics["harmonic_complexity"] = self.calculate_harmonic_complexity(midi_data)
        metrics["tempo_stability"] = self.calculate_tempo_stability(midi_data)
        metrics["dynamic_range"] = self.calculate_dynamic_range(midi_data)
        metrics["structural_coherence"] = self.calculate_structural_coherence(midi_data)

        # Additional basic metrics
        metrics["duration"] = midi_data.get_end_time()
        metrics["tempo"] = midi_data.estimate_tempo()
        metrics["total_notes"] = sum(len(instrument.notes) for instrument in midi_data.instruments)
        metrics["instrument_count"] = len(midi_data.instruments)

        return metrics

    def compare_metrics(
        self, metrics1: Dict[str, float], metrics2: Dict[str, float]
    ) -> Dict[str, float]:
        """Compare two sets of metrics."""
        comparison = {}

        for key in metrics1.keys():
            if key in metrics2:
                val1 = metrics1[key]
                val2 = metrics2[key]

                if val1 != 0 and val2 != 0:
                    # Calculate similarity as 1 - normalized difference
                    max_val = max(val1, val2)
                    similarity = 1 - abs(val1 - val2) / max_val
                    comparison[f"{key}_similarity"] = similarity
                else:
                    comparison[f"{key}_similarity"] = 0.0

        # Overall similarity
        similarities = [v for k, v in comparison.items() if k.endswith("_similarity")]
        if similarities:
            comparison["overall_similarity"] = np.mean(similarities)
        else:
            comparison["overall_similarity"] = 0.0

        return comparison

    def calculate_style_metrics(self, midi_data: pretty_midi.PrettyMIDI) -> Dict[str, Any]:
        """Calculate style-specific metrics."""
        style_metrics = {}

        # Jazz-like features
        all_notes = []
        for instrument in midi_data.instruments:
            all_notes.extend(instrument.notes)

        if all_notes:
            # Swing rhythm detection
            durations = [note.end - note.start for note in all_notes]
            short_durations = [d for d in durations if d < 0.5]
            if short_durations:
                swing_ratio = np.std(short_durations) / np.mean(short_durations)
                style_metrics["swing_characteristic"] = swing_ratio

            # Syncopation detection
            # (simplified: count notes that don't start on beat boundaries)
            syncopated_notes = 0
            for note in all_notes:
                beat_position = (note.start * 4) % 1  # Assuming 4/4 time
                if 0.1 < beat_position < 0.9:  # Not on strong beats
                    syncopated_notes += 1

            style_metrics["syncopation_ratio"] = syncopated_notes / len(all_notes)

        return style_metrics

    def get_metric_descriptions(self) -> Dict[str, str]:
        """Get descriptions of all metrics."""
        return {
            "note_density": "Number of notes per second",
            "velocity_diversity": "Standard deviation of note velocities",
            "pitch_range": "Difference between highest and lowest pitches",
            "rhythm_consistency": "Standard deviation of note durations",
            "melodic_contour": "Average pitch interval between consecutive notes",
            "harmonic_complexity": "Maximum number of simultaneous notes",
            "tempo_stability": "Standard deviation of tempo changes",
            "dynamic_range": "Difference between highest and lowest velocities",
            "structural_coherence": "Repetition pattern strength (autocorrelation)",
            "duration": "Total duration in seconds",
            "tempo": "Estimated tempo in BPM",
            "total_notes": "Total number of notes",
            "instrument_count": "Number of instruments used",
        }
