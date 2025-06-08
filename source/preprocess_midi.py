import mido
import math

# Constants from the paper (or sensible defaults if not specified)
TIME_RESOLUTION = 0.01  # 10ms steps for time shifts
MAX_TIME_SHIFT = 1.0    # Max time shift of 1 second
VELOCITY_BINS = 32
MAX_VELOCITY = 127

def quantize_time_shift(dt):
    """Quantizes time shift into discrete steps."""
    if dt > MAX_TIME_SHIFT:
        dt = MAX_TIME_SHIFT
    elif dt < 0:
        dt = 0 # Should not happen with mido delta times
    return min(int(round(dt / TIME_RESOLUTION)), int(MAX_TIME_SHIFT / TIME_RESOLUTION) -1) # 0 to 99 for 100 steps

def quantize_velocity(velocity):
    """Quantizes velocity into discrete bins."""
    if velocity <= 0:
        return 0
    if velocity > MAX_VELOCITY:
        velocity = MAX_VELOCITY
    # Ensure velocity is at least 1 for binning to work if we want 1-32 bins
    # The paper says 32 bins. Let's map 1-127 to 0-31.
    # A simple way: velocity / (MAX_VELOCITY / VELOCITY_BINS)
    # Or, if velocity 0 is a valid concept (e.g. note_off with velocity 0)
    # For SET_VELOCITY, it implies an active velocity.
    bin_size = (MAX_VELOCITY + 1) / VELOCITY_BINS # +1 to include 127 in the last bin properly
    return min(int(velocity / bin_size), VELOCITY_BINS - 1)

def midi_to_event_sequence(midi_path):
    """
    Converts a MIDI file to a sequence of symbolic events.
    Events: NOTE_ON_p, NOTE_OFF_p, TIME_SHIFT_dt, SET_VELOCITY_v
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        print(f"Error loading MIDI file {midi_path}: {e}")
        return None

    event_sequence = []
    current_time_absolute = 0
    last_event_time_absolute = 0
    current_velocity_bin = quantize_velocity(64) # Default velocity

    # First, set initial velocity
    event_sequence.append(f"SET_VELOCITY_{current_velocity_bin}")

    # Collect all note events with their absolute times and track info
    timed_events = []
    for i, track in enumerate(mid.tracks):
        absolute_time_in_track = 0
        for msg in track:
            absolute_time_in_track += msg.time
            if msg.type in ["note_on", "note_off"]:
                timed_events.append({
                    "time": absolute_time_in_track,
                    "type": msg.type,
                    "note": msg.note,
                    "velocity": msg.velocity,
                    "track": i
                })
            elif msg.type == "control_change" and msg.control == 7: # Main volume, could be used for velocity context
                # This is a simplification; actual velocity handling is per-note
                pass # For now, we only use note_on velocities

    # Sort all events by time, then by type (note_off before note_on for same time)
    timed_events.sort(key=lambda x: (x["time"], 0 if x["type"] == "note_off" else 1))

    for msg_event in timed_events:
        event_abs_time = msg_event["time"]
        delta_time = event_abs_time - last_event_time_absolute

        if delta_time > 0:
            quantized_dt = quantize_time_shift(delta_time)
            event_sequence.append(f"TIME_SHIFT_{quantized_dt}")

        if msg_event["type"] == "note_on" and msg_event["velocity"] > 0:
            # Check if velocity changed significantly enough to warrant a SET_VELOCITY event
            new_velocity_bin = quantize_velocity(msg_event["velocity"])
            if new_velocity_bin != current_velocity_bin:
                current_velocity_bin = new_velocity_bin
                event_sequence.append(f"SET_VELOCITY_{current_velocity_bin}")
            event_sequence.append(f"NOTE_ON_{msg_event['note']}")
        elif msg_event["type"] == "note_off" or (msg_event["type"] == "note_on" and msg_event["velocity"] == 0):
            event_sequence.append(f"NOTE_OFF_{msg_event['note']}")
        
        last_event_time_absolute = event_abs_time

    return event_sequence

if __name__ == "__main__":
    # Test with a sample MIDI file from the paired data
    # Load the paired data to get a MIDI path
    import json
    paired_data_path = "/home/ubuntu/paired_midi_text_sample.json"
    try:
        with open(paired_data_path, "r") as f:
            paired_data = json.load(f)
        if not paired_data:
            print("No data in paired_midi_text_sample.json")
            exit()
        
        sample_midi_path = paired_data[0]["file_path"]
        print(f"Processing MIDI file: {sample_midi_path}")
        event_seq = midi_to_event_sequence(sample_midi_path)

        if event_seq:
            print(f"Generated event sequence (first 50 events):\n{event_seq[:50]}")
            # Save the sequence for inspection if needed
            output_event_file = "/home/ubuntu/sample_midi_events.txt"
            with open(output_event_file, "w") as f_out:
                for event in event_seq:
                    f_out.write(event + "\n")
            print(f"Full event sequence saved to {output_event_file}")

    except FileNotFoundError:
        print(f"Could not find {paired_data_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

