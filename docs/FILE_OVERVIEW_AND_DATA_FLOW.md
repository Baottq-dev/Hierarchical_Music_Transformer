# 🗂️ Tổng Quan File & Luồng Dữ Liệu AMT

> Tài liệu này liệt kê **toàn bộ cấu trúc tệp** trong dự án AMT và mô tả chi tiết **luồng dữ liệu** từ đầu vào thô → đầu ra nhạc sinh.

---

## 1. Cây thư mục (rút gọn)
```
AMT/
 ├─ data/
 │   ├─ midi/                # Bộ MIDI gốc (Lakh MIDI Clean)
 │   ├─ text/                # Mô tả văn bản (tùy chọn tự nhập)
 │   ├─ processed/           # Kết quả tiền xử lý (token, v.v.)
 │   ├─ output/              # JSON trung gian (metadata, embedding, clustering)
 │   ├─ reference/           # MIDI tham chiếu để đánh giá
 │   └─ evaluation/          # Báo cáo metric sau sinh
 ├─ models/
 │   └─ checkpoints/         # GPT-2 checkpoint (*.pt)
 ├─ source/
 │   ├─ data_collection/
 │   ├─ data_processing/
 │   ├─ model/
 │   ├─ evaluation/
 │   ├─ utils/
 │   └─ scripts/
 ├─ paper/
 ├─ requirements.txt
 ├─ README.md
 ├─ PROJECT_REPORT.md
 └─ FILE_OVERVIEW_AND_DATA_FLOW.md  <-- (tài liệu này)
```

## 2. Mô tả chi tiết từng thư mục & file chính
### 2.1 `source/data_collection/`
| File | Chức năng |
|------|-----------|
| **midi_metadata.py** | Quét thư mục MIDI, sinh JSON `{file_path, artist, title}`; hỗ trợ CLI. |
| **wikipedia_collector.py** | Gọi Wikipedia API, ghép miêu tả với mỗi MIDI; lưu JSON. |
| **__init__.py** | Khai báo package. |

### 2.2 `source/data_processing/`
| File | Chức năng |
|------|-----------|
| **midi_processor.py** | Phân tích & chuyển MIDI ↔︎ chuỗi sự kiện; thống kê đặc trưng. |
| **text_processor.py** | Làm sạch văn bản, trích keyword, BERT embedding, hàm batch `get_bert_embeddings`. |
| **collect_text.py** | Demo thu thập mô tả (placeholder). |
| **process_midi.py / process_text.py** | Script nhỏ chạy hàm trong `*_processor.py`. |
| **prepare_training.py** | Treo móc sang `utils/data_preparation.py`. |
| **__init__.py** | Export hàm tiện dụng. |

### 2.3 `source/model/`
| File | Chức năng |
|------|-----------|
| **training.py** | Định nghĩa `AMTModel`, dataset, vòng huấn luyện & checkpoint. |
| **generation.py** | Hai lớp `MusicGenerator` (token-based) & `AMTGenerator` (BERT→GPT-2) để sinh nhạc. |
| **clustering.py** | K-means cho embedding, gán `semantic_token`. |
| **__init__.py** | Khai báo package. |

### 2.4 `source/evaluation/`
| File | Chức năng |
|------|-----------|
| **metrics.py** | 5 hàm metric + hàm batch; xuất `overall_score`. |
| **__init__.py** | Export metric list. |

### 2.5 `source/utils/`
| File | Chức năng |
|------|-----------|
| **data_preparation.py** | Ghép `semantic_token` + chuỗi sự kiện → training JSON. |
| **environment.py** | In version lib. |
| **__init__.py** | Shortcuts. |

### 2.6 `source/scripts/`
| File | Chức năng |
|------|-----------|
| **main.py** | Orchestrator End-to-End (5 bước) – entry point đề xuất. |
| **__init__.py** | Khai báo package. |

### 2.7 File cấu hình gốc
| File | Mô tả |
|------|-------|
| **config.py** | Chứa toàn bộ tham số (data path, model, training, generation, eval). |

---

## 3. Luồng dữ liệu chi tiết (tên file sinh kèm đường dẫn)
1. **Quét MIDI**  
   ```bash
   python source/data_collection/midi_metadata.py data/midi data/output/midi_metadata_list.json
   ```
   • Kết quả: `data/output/midi_metadata_list.json`

2. **Ghép mô tả Wikipedia**  
   ```bash
   python source/data_collection/wikipedia_collector.py \
          data/output/midi_metadata_list.json \
          data/output/automated_paired_data.json
   ```
   • Kết quả: `.../automated_paired_data.json` (thêm `text_description`).

3. **Sinh BERT embedding**  
   `scripts/main.py` gọi `text_processor.get_bert_embeddings` → ghi `data/output/text_embeddings.json`:
   ```json
   {
     "file_path": "...mid",
     "artist": "Abba",
     "title": "Mamma Mia",
     "text_description": "...",
     "embedding": [0.12, 0.03, ...]
   }
   ```

4. **Gom cụm & gán token**  
   `model/clustering.py` → `data/output/clustered_text_data.json`:
   ```json
   {
     "file_path": "...",
     "semantic_token": 3,
     ...
   }
   ```

5. **Chuẩn bị dữ liệu huấn luyện**  
   `utils/data_preparation.py` → `data/output/amt_training_data.json`:
   ```json
   {
    "midi_event_sequence": [[t,n,d], ...],
    "semantic_token_str":"SEMANTIC_TOKEN_3",
    "combined_sequence_for_amt": ["SEMANTIC_TOKEN_3", [t1,n1,d1], ...]
   }
   ```

6. **Huấn luyện**  
   ```bash
   python source/model/training.py   # Đọc amt_training_data.json
   ```
   • Tạo checkpoint `models/checkpoints/checkpoint_epoch_X.pt`.

7. **Sinh nhạc**  
   ```python
   from AMT.source.model.generation import AMTGenerator
   gen = AMTGenerator("models/checkpoints/checkpoint_epoch_10.pt")
   gen.generate_music("Bản piano buồn", "output/generated/my_song.mid")
   ```

8. **Đánh giá**  
   ```python
   from AMT.source.evaluation.metrics import evaluate_generated_music
   evaluate_generated_music("reference/ref.mid", "output/generated/my_song.mid")
   ```
   • Ghi JSON/txt vào `data/evaluation/` nếu cần.

---

## 4. Tương quan gọi hàm (call-graph rút gọn)
```text
scripts/main.py
 ├─ midi_metadata.list_midi_files_and_metadata
 ├─ wikipedia_collector.pair_midi_with_wikipedia
 ├─ text_processor.get_bert_embeddings
 ├─ clustering.cluster_embeddings
 └─ data_preparation.prepare_training_data
```

## 5. Phụ thuộc & yêu cầu môi trường
Đã liệt kê tại `requirements.txt`. Chạy thử:
```bash
python -m AMT.source.utils.environment
```

---
**File này nhằm hỗ trợ developer mới hiểu nhanh cấu trúc & hành trình của dữ liệu.** 