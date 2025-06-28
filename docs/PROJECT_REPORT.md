# 📑 BÁO CÁO DỰ ÁN  
**The Beat Goes On – Symbolic Music Generation with Text Controls (AMT)**  

---

## 1. Giới thiệu tổng quan
AMT (Audio Music Transformer) là hệ thống sinh nhạc tự động từ mô tả văn bản. Hệ thống kết hợp **BERT** (hiểu ngôn ngữ tự nhiên) và **GPT-2** (mô hình hoá chuỗi thời gian) để tạo ra **chuỗi sự kiện MIDI** phù hợp với nội dung, cảm xúc và phong cách mà người dùng yêu cầu.

Mục tiêu chính:
1. Tự động **thu thập** dữ liệu MIDI & mô tả văn bản.
2. **Tiền xử lý** & chuyển đổi dữ liệu thành định dạng huấn luyện.
3. **Huấn luyện** mô hình kết hợp BERT–GPT-2.
4. **Sinh** nhạc mới dựa trên prompt văn bản.
5. **Đánh giá** chất lượng nhạc sinh bằng bộ chỉ số định lượng.

## 2. Bộ dữ liệu
| Thành phần | Nguồn | Quy mô |
|------------|-------|--------|
| Lakh MIDI Clean | Raffel et al. | ≈ 136 k file MIDI (hơn 2 000 thư mục nghệ sĩ) |
| Mô tả Wikipedia | API _wikipedia_ | 1 mô tả/bài | 

Sau khi quét, hệ thống lưu siêu dữ liệu vào `AMT/data/output/*` ở định dạng JSON.

## 3. Kiến trúc hệ thống
```mermaid
graph TD
    A[Lakh MIDI Clean] --> B[MIDI Processing]
    C[Wikipedia] --> D[Text Processing]
    B --> E[MIDI Event Seq]
    D --> F[BERT Embedding]
    E --> G[Training Data (JSON)]
    F --> G
    G --> H[Huấn luyện AMT]
    H --> I[GPT-2 + Projection]
    subgraph Sinh & Đánh giá
      I --> J[Sinh Nhạc]
      J --> K[MIDI Output]
      K --> L[Evaluation Metrics]
    end
```

### Thư mục mã nguồn
```
AMT/
 ├─ data/              # dữ liệu
 ├─ source/
 │   ├─ data_collection/   # thu thập
 │   ├─ data_processing/   # tiền xử lý
 │   ├─ model/             # huấn luyện & sinh
 │   ├─ evaluation/        # đánh giá
 │   ├─ utils/             # hàm hỗ trợ
 │   └─ scripts/           # main.py (pipeline)
 ├─ models/            # checkpoint GPT-2
 └─ README.md
```

## 4. Pipeline xử lý dữ liệu
| Bước | Script | Mô tả |
|------|--------|-------|
| 1 | `data_collection/midi_metadata.py` | Quét thư mục MIDI → JSON `{path, artist, title}` |
| 2 | `data_collection/wikipedia_collector.py` | Gọi Wikipedia → lấy mô tả, URL |
| 3 | `data_processing/text_processor.py` | Làm sạch + BERT embedding + trích keyword |
| 4 | `model/clustering.py` | K-means trên embedding → gán **semantic_token** |
| 5 | `utils/data_preparation.py` | Ghép `semantic_token` + chuỗi sự kiện MIDI → training data |

## 5. Mô hình huấn luyện
- **Projection**: Linear 768 → 1024 để đưa embedding BERT vào không gian GPT-2.
- **GPT-2**: 6 lớp, 8 head, 1 024 hidden dim, vocab 512 (token hóa sự kiện MIDI).
- **Loss**: Language-modeling (Cross-entropy) trên chuỗi `[semantic_token] + event_seq`.
- **Checkpoint**: lưu mỗi epoch vào `models/checkpoints/`.

## 6. Sinh nhạc
Hàm chính `AMTGenerator.generate_music()`:
1. BERT embedding cho mô tả.
2. Projection → GPT-2.generate (top-k, top-p, temperature).
3. Giải mã token → (TIME_ON, NOTE, DUR) → `event_sequence_to_midi()`.
4. Xuất `.mid` vào `output/generated/`.

## 7. Đánh giá
| Metric | Ý nghĩa |
|--------|---------|
| Note Density Ratio | Tỉ lệ mật độ nốt giữa gốc & sinh |
| Velocity Similarity | Sai khác mean/std vận tốc |
| Note Range Similarity | Jaccard trên quãng nốt |
| Time-signature Match | Trùng nhịp |
| Tempo Similarity | Tương đồng BPM |
| **Overall** | Trung bình có trọng số |

## 8. Kết quả thử nghiệm (demo)
| Prompt | Overall Score |
|--------|---------------|
| "Bản ballad piano nhẹ nhàng, cảm xúc hứng khởi" | **0.82** |
| "Bản rock guitar mạnh mẽ, tiết tấu nhanh" | 0.76 |

*(Số liệu giả lập để minh hoạ)*

## 9. Cải tiến đã triển khai
- Thêm hàm **`get_bert_embeddings`** hỗ trợ batch.
- Xoá script dư `run_pipeline.py`, _extract_midi_metadata.py_.
- Chuyển dataset vào `AMT/data/`, dọn file rác.
- README cập nhật hướng dẫn, thêm tải NLTK/SpaCy.

## 10. Hướng phát triển
1. **Tokenization nâng cao**: chord, velocity, tempo events riêng.
2. **Data Augmentation**: transpose, humanize timing.
3. **GUI Web**: nhập prompt, nghe nhạc trực tuyến.
4. **Fine-tune đánh giá**: thêm Fréchet Audio Distance cho symbolic.
5. **Multi-track / style transfer**: điều khiển đa nhạc cụ.

## 11. Cài đặt & chạy nhanh
```bash
cd AMT
python -m venv venv && venv\Scripts\activate    # Windows
pip install -r requirements.txt
python -m nltk.downloader punkt averaged_perceptron_tagger
python -m spacy download en_core_web_sm

# Pipeline end-to-end
python source/scripts/main.py
```

## 12. Tài liệu tham khảo
- Raffel et al., "Learning-based methods for expressive performance…" (Lakh MIDI).  
- Vaswani et al., "Attention is All You Need".  
- Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers…".  
- Radford et al., "Language Models are Unsupervised Multitask Learners".

---
**© 2025 FPT – TMG301** 