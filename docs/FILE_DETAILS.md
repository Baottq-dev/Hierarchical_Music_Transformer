# 📚 Chi Tiết Nội Dung Từng File – Dự Án AMT

> Tài liệu này mô tả **chức năng chi tiết** của từng tệp quan trọng trong dự án `AMT`. Các file được sắp xếp theo thư mục để tiện tra cứu. Những tệp không ảnh hưởng trực tiếp (cache, \_\_pycache\_\_, file dữ liệu) được bỏ qua.

---

## 1. Gốc dự án
| File/Thư mục | Mô tả |
|--------------|-------|
| `requirements.txt` | Liệt kê thư viện Python cần cài đặt. |
| `README.md` | Hướng dẫn cài đặt, chạy, kiến trúc tổng thể. |
| `PROJECT_REPORT.md` | Báo cáo tổng quan dự án (mục tiêu, mô hình, kết quả). |
| `FILE_OVERVIEW_AND_DATA_FLOW.md` | Sơ đồ cây thư mục & luồng dữ liệu. |
| `FILE_DETAILS.md` | (tệp hiện tại) bảng chú giải chi tiết từng tệp. |
| `paper/` | Chứa các bài báo tham khảo. |

---

## 2. Thư mục `AMT/data`
| Thư mục con | Mục đích |
|--------------|----------|
| `midi/` | Lakh MIDI Clean (input). |
| `text/` | Text mô tả thô (nếu thu thập thủ công). |
| `processed/` | Dữ liệu đã token hoá / vector hoá. |
| `output/` | Các JSON trung gian (`*_metadata.json`, `*_embeddings.json`, ...). |
| `reference/` | MIDI mẫu để đánh giá. |
| `evaluation/` | Báo cáo metric chi tiết sau khi sinh nhạc. |

---

## 3. Thư mục `AMT/source`

### 3.1 `config.py`
- **Loại:** Module cấu hình.
- **Nội dung chính:** Hằng số về dữ liệu, model, training, generation và evaluation; hàm `get_config()` tổng hợp tất cả.

### 3.2 `__init__.py`
- Khai báo namespace `AMT.source` export các sub-package `data_processing`, `model`, `evaluation`, `config`.

---

### 3.3 `data_collection/`
| File | Mô tả chi tiết |
|------|----------------|
| `midi_metadata.py` | • Hàm `list_midi_files_and_metadata(base_dir)` duyệt đệ quy thư mục, trích `artist` theo tên thư mục, `title` theo tên file.  <br>• Hàm `save_metadata(...)` ghi JSON. <br>• Khối `main()` (argparse) để chạy CLI. |
| `wikipedia_collector.py` | • Hàm `get_wikipedia_summary(artist,title)` tìm page qua `wikipedia` pkg. <br>• Hàm `pair_midi_with_wikipedia(metadata_file, output_file)` đọc JSON metadata, gọi API, sleep giữa request, ghi `automated_paired_data.json`. <br>• Sử dụng regex làm sạch, xử lý PageError, DisambiguationError. |
| `__init__.py` | Gán alias, không chứa logic. |

---

### 3.4 `data_processing/`
| File | Mô tả chi tiết |
|------|----------------|
| `midi_processor.py` | **Trái tim symbolic processing**. <br>• Hằng số TIME_RESOLUTION, MAX_TIME_SHIFT,... <br>• Hàm `get_midi_metadata` (trích meta track, tempo, time sig). <br>• Hàm `midi_to_event_sequence` & `event_sequence_to_midi`. <br>• Hàm `analyze_midi_file` (tính density, range, velocity, tempo). |
| `text_processor.py` | • Hàm `clean_text`, `extract_music_keywords`, `extract_keywords` (TF-IDF). <br>• Hàm `get_text_features` trả dict thống kê + keyword. <br>• Hàm `get_bert_embedding` (tokenizer & model BERT-base) & `get_bert_embeddings` (batch). <br>• Hàm `scrape_wikipedia` (dự phòng). <br>• Hàm `process_text_descriptions` thống kê toàn tập. <br>• Hàm `create_training_examples`. |
| `collect_text.py` | Placeholder thu text (ít logic). |
| `process_midi.py` | Gọi `midi_processor` trên toàn bộ thư mục rồi xuất thống kê (demo). |
| `process_text.py` | Gọi `text_processor.process_text_descriptions`. |
| `prepare_training.py` | Wrapper gọi `utils/data_preparation`. |
| `__init__.py` | Xuất hằng/func tiện dụng. |

---

### 3.5 `model/`
| File | Chi tiết |
|------|---------|
| `training.py` | • Class `AMTDataset` đọc `amt_training_data.json`. <br>• Class `AMTModel`: Linear projection 768→1024 + GPT-2 (6 layers). <br>• Hàm `train_model` chạy DataLoader, Adam, checkpoint. <br>• `__main__` -> parse path mặc định & huấn luyện. |
| `generation.py` | • Class `MusicGenerator` dùng tokenizer HF + GPT-2 fine-tuned. <br>• Class `AMTGenerator`: load checkpoint, sinh sequence từ text embedding, convert event→MIDI. <br>• Hàm `load_generator`, `save_generated_sequences`. |
| `clustering.py` | • Hàm `determine_optimal_k`, `cluster_embeddings`. <br>• Class `MIDIClusterer` (phân cụm theo đặc trưng âm nhạc). <br>• Hàm `cluster_midi_files` convenience. |
| `__init__.py` | Exports. |

---

### 3.6 `evaluation/`
| File | Nội dung |
|------|----------|
| `metrics.py` | Định nghĩa 5 metric (density, velocity, range, time-sig, tempo) + `evaluate_generated_music` và `evaluate_batch`. |
| `__init__.py` | Import metric list. |

---

### 3.7 `utils/`
| File | Nội dung |
|------|----------|
| `data_preparation.py` | Đọc `clustered_text_data.json`, ghép SEMANTIC_TOKEN_i + chuỗi event, lưu `amt_training_data.json`. <br>• Log tiến độ, xử lý lỗi file mất. |
| `environment.py` | In version `torch`, `transformers`, `mido`, ... để debug môi trường. |
| `__init__.py` | Shortcut import. |

---

### 3.8 `scripts/`
| File | Nội dung |
|------|----------|
| `main.py` | Pipeline "one-click": verify env → metadata → wiki → embedding → clustering → prepare training. |
| `__init__.py` | Dummy. |

---

## 4. Thư mục `models/`
- Trống ban đầu, sẽ chứa `checkpoints/checkpoint_epoch_*.pt` do `training.py` sinh.

## 5. Thư mục `paper/`
- Tài liệu tham khảo PDF ("The Beat Goes On-...", "Anticipatory Music Transformer", ...).

---

## 6. Liên hệ giữa file
- **data_collection** sinh JSON đầu vào cho **data_processing/text_processor**.  
- **text_processor** + **clustering** cung cấp semantic_token cho **utils/data_preparation**.  
- **training.py** tiêu thụ output đó, sinh checkpoint cho **generation.py**.  
- **generated midi** + **reference midi** đi vào **evaluation/metrics.py**.

---
**Tài liệu hoàn tất – mọi file quan trọng đều được mô tả.** 