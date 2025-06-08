## AMT: Automatic Music Tokenization & Generation

### Mục tiêu dự án
Dự án xây dựng pipeline tự động từ dữ liệu MIDI và mô tả văn bản, qua các bước:
1. Thu thập metadata và mô tả từ Wikipedia cho file MIDI.
2. Sinh embedding văn bản, phân cụm thành các semantic token.
3. Chuyển đổi MIDI thành chuỗi sự kiện (event sequence).
4. Chuẩn bị dữ liệu huấn luyện cho mô hình sinh nhạc (fine-tune GPT-2).
5. Sinh nhạc mới dựa trên semantic token.

---

### Cấu trúc thư mục
```
source/
  automate_wikipedia_pairing.py      # Ghép metadata MIDI với mô tả Wikipedia
  collect_text_data_conceptual.py    # Kịch bản ý tưởng thu thập mô tả văn bản
  cluster_text_embeddings.py         # Phân cụm embedding văn bản thành semantic token
  extract_midi_metadata.py           # Quét thư mục MIDI, sinh metadata
  extract_text_features.py           # Sinh embedding từ mô tả văn bản (BERT)
  fine_tune_amt.py                   # Huấn luyện/fine-tune mô hình sinh nhạc
  generate_music.py                  # Sinh nhạc mới từ semantic token
  prepare_amt_training_data.py       # Chuẩn bị dữ liệu huấn luyện từ MIDI & semantic token
  preprocess_midi.py                 # Chuyển MIDI thành event sequence
  verify_env.py                      # Kiểm tra môi trường/thư viện
data/
  output/
    midi_metadata_list.json          # Metadata các file MIDI
    automated_paired_data_1000.json  # Dữ liệu ghép tự động (MIDI + mô tả)
    midi_metadata_list.json          # Metadata chi tiết các file MIDI
    ... (các file output khác)
.vscode/
  settings.json                      # Cấu hình VSCode
requirement.txt                      # Thư viện phụ thuộc
README.md                            # Tài liệu này
```

---

### Hướng dẫn cài đặt

1. **Cài đặt thư viện phụ thuộc:**
   ```
   pip install -r requirement.txt
   ```
   Các thư viện chính: `torch`, `transformers`, `mido`, `numpy`, `scikit-learn`, `wikipedia`, `matplotlib`.

2. **Kiểm tra môi trường:**
   ```
   python source/verify_env.py
   ```

---

### Pipeline sử dụng

1. **Trích xuất metadata từ thư mục MIDI:**
   ```
   python source/extract_midi_metadata.py
   ```
   - Sinh file `data/output/midi_metadata_list.json`.

2. **Ghép metadata với mô tả Wikipedia:**
   ```
   python source/automate_wikipedia_pairing.py
   ```
   - Sinh file `data/output/automated_paired_data.json`.

3. **Sinh embedding từ mô tả văn bản:**
   ```
   python source/extract_text_features.py
   ```
   - Sinh file `data/output/text_embeddings.json`.

4. **Phân cụm embedding thành semantic token:**
   ```
   python source/cluster_text_embeddings.py
   ```
   - Sinh file `data/output/clustered_text_data.json`.

5. **Chuẩn bị dữ liệu huấn luyện cho mô hình sinh nhạc:**
   ```
   python source/prepare_amt_training_data.py
   ```
   - Sinh file `data/output/amt_training_data.json`.

6. **Fine-tune mô hình sinh nhạc (GPT-2):**
   ```
   python source/fine_tune_amt.py
   ```
   - Lưu mô hình vào `data/output/amt_model_fine_tuned/`.

7. **Sinh nhạc mới từ semantic token:**
   ```
   python source/generate_music.py
   ```
   - Kết quả MIDI sinh ra trong `data/output/generated_music/`.

---

### Ghi chú

- Các script có thể cần điều chỉnh đường dẫn dữ liệu cho phù hợp với hệ thống của bạn.
- Một số script chỉ là ý tưởng/kịch bản (ví dụ: `collect_text_data_conceptual.py`), cần thực hiện bằng các công cụ hoặc API thực tế.
- Chất lượng nhạc sinh ra phụ thuộc vào dữ liệu và quá trình fine-tune.
