import os

def count_midi_files(directory):
    midi_count = 0
    artists_with_files = 0
    artist_files = {}
    
    for root, dirs, files in os.walk(directory):
        midi_files_in_dir = 0
        
        for file in files:
            if file.lower().endswith(('.mid', '.midi')):
                midi_count += 1
                midi_files_in_dir += 1
        
        # Thêm thông tin cho thư mục hiện tại (chỉ thư mục con trực tiếp của thư mục gốc)
        relative_path = os.path.relpath(root, directory)
        if os.path.dirname(relative_path) == '':  # Chỉ xét thư mục con cấp 1
            if midi_files_in_dir > 0:
                artists_with_files += 1
                artist_name = os.path.basename(root)
                artist_files[artist_name] = midi_files_in_dir
    
    # Sắp xếp nghệ sĩ theo số lượng file MIDI
    sorted_artists = sorted(artist_files.items(), key=lambda x: x[1], reverse=True)
    
    print(f"Tổng số file MIDI: {midi_count}")
    print(f"Số nghệ sĩ có file MIDI: {artists_with_files}")
    print("\nTop 10 nghệ sĩ có nhiều file MIDI nhất:")
    for i, (artist, count) in enumerate(sorted_artists[:10], 1):
        print(f"{i}. {artist}: {count} file")
    
    print("\nPhân bố số lượng file:")
    ranges = [0, 1, 5, 10, 20, 50, 100, float('inf')]
    range_labels = ['0', '1-4', '5-9', '10-19', '20-49', '50-99', '100+']
    distribution = [0] * (len(ranges) - 1)
    
    for count in artist_files.values():
        for i in range(len(ranges) - 1):
            if ranges[i] <= count < ranges[i+1]:
                distribution[i] += 1
                break
    
    for i, label in enumerate(range_labels):
        print(f"{label} file: {distribution[i]} nghệ sĩ")

if __name__ == "__main__":
    midi_directory = "data/midi"  # Đường dẫn đến thư mục MIDI
    count_midi_files(midi_directory)