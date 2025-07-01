# 02_DATA_PROCESSING_REPORT.md

## 1. Mục tiêu
Xử lý, chuẩn hóa, trích xuất đặc trưng từ MIDI và text, chuẩn bị dữ liệu cho huấn luyện mô hình.

## 2. Logic tổng thể
- Đọc paired data (MIDI-text).
- Xử lý MIDI đa track: mapping nhạc cụ, trích xuất event sequence, metadata, đặc trưng (note, velocity, tempo, ...).
- Xử lý text: chuẩn hóa, embedding (BERT, TF-IDF, spaCy), trích xuất cảm xúc, thể loại, nhạc cụ.
- Kết hợp dữ liệu, chia train/val/test, chuẩn hóa batch.
- Xuất file processed/training data.

## 3. Class chính
- `MIDIProcessor`: Xử lý, trích xuất đặc trưng MIDI.
- `TextProcessor`: Xử lý, embedding, trích xuất đặc trưng text.
- `DataPreparer`: Kết hợp, chia data, tạo batch, chuẩn hóa.

## 4. Input/Output
- **Input:** File paired data (`data/output/complete_dataset.json` hoặc tương đương).
- **Output:**
  - `data/processed/processed_data.json`
  - `data/processed/training_data.json`

## 5. Flow chi tiết
1. Đọc paired data.
2. Xử lý từng MIDI: mapping, trích xuất event, đặc trưng.
3. Xử lý từng text: chuẩn hóa, embedding, trích xuất đặc trưng.
4. Kết hợp, chia train/val/test, tạo batch.
5. Xuất file JSON.

## 6. Điểm mạnh
- Hỗ trợ MIDI đa track, mapping nhạc cụ, trích xuất metadata nâng cao.
- Xử lý text đa dạng: embedding, cảm xúc, thể loại, nhạc cụ.
- Chuẩn hóa dữ liệu, chia batch tối ưu cho huấn luyện.

## 7. Hướng mở rộng
- Tích hợp thêm feature extraction (chord, key, structure).
- Hỗ trợ nhiều loại embedding text.
- Tối ưu hóa tốc độ xử lý, memory. 