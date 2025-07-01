# 01_DATA_COLLECTION_REPORT.md

## 1. Mục tiêu
Thu thập dữ liệu MIDI, metadata, lyrics, text mô tả, ghép cặp dữ liệu, lọc chất lượng, chuẩn hóa đầu vào cho pipeline.

## 2. Logic tổng thể
- Quét thư mục MIDI, trích xuất metadata (tên, tempo, nhạc cụ, độ dài, ...).
- Thu thập text mô tả (lyrics, mô tả, cảm xúc, thể loại) từ file hoặc nguồn ngoài.
- Ghép cặp MIDI với text phù hợp.
- Lọc dữ liệu theo chất lượng (độ dài text, thời lượng MIDI, ...).
- Xuất file JSON: metadata, paired data, complete dataset.

## 3. Class chính
- `MIDICollector`: Quét, trích xuất metadata từ MIDI.
- `TextCollector`: Thu thập, chuẩn hóa text mô tả.
- `DataPairing`: Ghép cặp, lọc, validate dữ liệu.

## 4. Input/Output
- **Input:** Thư mục MIDI (`data/midi/`), text/lyrics (`data/text/`), metadata.
- **Output:**
  - `data/output/midi_metadata.json`
  - `data/output/paired_data.json`
  - `data/output/complete_dataset.json`

## 5. Flow chi tiết
1. Khởi tạo collector, quét MIDI, trích xuất metadata.
2. Thu thập text mô tả cho từng MIDI.
3. Ghép cặp, lọc dữ liệu, validate.
4. Xuất file JSON.

## 6. Điểm mạnh
- Tự động hóa, dễ mở rộng nguồn dữ liệu.
- Lọc chất lượng, validate dữ liệu đầu vào.
- Dễ dàng tích hợp thêm metadata, text đặc trưng mới.

## 7. Hướng mở rộng
- Crawl lyrics/text tự động từ web.
- Tích hợp thêm metadata (genre, mood, composer, ...).
- Hỗ trợ nhiều định dạng nhạc (MusicXML, audio, ...). 