# 🔄 DATA FLOW – Dự Án AMT

> Tài liệu này mô tả **luồng dữ liệu** chi tiết nhất từ dữ liệu thô (MIDI & mô tả) tới đầu ra nhạc sinh và đánh giá.
>
> Các bước ghi rõ: **đầu vào**, **script/hàm thực thi**, **đầu ra** (tên file + đường dẫn).

---

## 1. Sơ đồ tổng quát (Mermaid)
```mermaid
graph TD
    %% Nguồn dữ liệu
    A[Lakh MIDI Clean (\ndata/midi)] --> B1[midi_metadata.py]
    W[Wikipedia API] --> B2[wikipedia_collector.py]

    %% Trung gian
    B1 --> C1[midi_metadata_list.json]
    B2 --> C2[automated_paired_data.json]

    %% Embed & clustering
    C2 --> D1[get_bert_embeddings]
    D1 --> E1[text_embeddings.json]
    E1 --> F1[cluster_embeddings]
    F1 --> G1[clustered_text_data.json]

    %% Ghép với MIDI events
    G1 & A --> H1[midi_to_event_sequence]
    H1 --> I1[data_preparation.py]
    I1 --> J1[amt_training_data.json]

    %% Huấn luyện
    J1 --> K1[training.py]
    K1 --> L1[GPT-2 checkpoint]

    %% Sinh & đánh giá
    L1 --> M1[AMTGenerator.generate_music]
    M1 --> N1[generated.mid]
    A --> Ref[reference.mid]
    N1 & Ref --> O1[evaluation.metrics]
    O1 --> P1[eval_report.json]
```

---

## 2. Bảng tuần tự (Input → Output)
| # | Bước | Script/Hàm chính | Đầu vào | Đầu ra |
|---|------|------------------|---------|--------|
| 1 | Liệt kê MIDI | `data_collection/midi_metadata.py` <br> `list_midi_files_and_metadata()` | `data/midi/**/*.mid` | `data/output/midi_metadata_list.json` |
| 2 | Ghép Wikipedia | `data_collection/wikipedia_collector.py` <br> `pair_midi_with_wikipedia()` | File JSON bước 1 | `data/output/automated_paired_data.json` (thêm `text_description`) |
| 3 | Text Embedding | `data_processing/text_processor.py` <br> `get_bert_embeddings()` | JSON bước 2 | `data/output/text_embeddings.json` (thêm `embedding`) |
| 4 | Clustering | `model/clustering.py` <br> `cluster_embeddings()` | JSON bước 3 | `data/output/clustered_text_data.json` (thêm `semantic_token`) |
| 5 | Chuỗi sự kiện + Token | `utils/data_preparation.py` <br> `midi_to_event_sequence()` | MIDI gốc + JSON bước 4 | `data/output/amt_training_data.json` (chứa `combined_sequence_for_amt`) |
| 6 | Huấn luyện | `model/training.py` <br> `train_model()` | Training JSON | `models/checkpoints/checkpoint_epoch_*.pt` |
| 7 | Sinh nhạc | `model/generation.py` <br> `AMTGenerator.generate_music()` | Checkpoint + prompt text | `output/generated/*.mid` |
| 8 | Đánh giá | `evaluation/metrics.py` <br> `evaluate_generated_music()` | MIDI gốc tham chiếu + MIDI sinh | `data/evaluation/eval_report.json` |

---

## 3. Chi tiết từng khối
### 3.1 Liệt kê & ghi metadata
- **Function:** `list_midi_files_and_metadata(base_dir)`  
- **Logic:** duyệt đệ quy, bắt tên nghệ sĩ từ `os.path.basename(root)`, rửa tên bài hát (`_` → space, xoá số đuôi).  
- **Tần suất chạy:** một lần chuẩn bị dữ liệu.

### 3.2 Ghép mô tả Wikipedia
- **Thuật toán tìm trang:** thử chuỗi tìm kiếm theo thứ tự: "`artist title (song)`", "…".  
- Dùng thư viện `wikipedia`; bắt lỗi Page/Disambiguation.

### 3.3 Tính embedding BERT
- **Model:** `bert-base-uncased` (Transformers).  
- **Truncate:** `max_length=512` token, lấy embedding token `[CLS]`.

### 3.4 Clustering semantic token
- **Embedding matrix:** N × 768.  
- **K determination:** silhouette score (min_k=2, max_k ≤ 10).  
- **KMeans:** `sklearn.cluster.KMeans` with `n_init="auto"`.

### 3.5 Ghép token & sự kiện MIDI
- **Trích event:** (TIME_ON, NOTE, DUR) với lượng tử 480 ticks.  
- **Semantic token:** format string `SEMANTIC_TOKEN_i` đệm đầu chuỗi.  
- **Output:** list JSON item chứa cả vector embedding (để huấn luyện đa phương thức).

### 3.6 Huấn luyện mô hình
- **Dataset:** mỗi sample → tuple `(text_embedding, event_sequence)`.
- **Model:** projection + GPT-2 (6L, 8H, 1 024H, vocab 512).  
- **Loss:** tự động qua `GPT2LMHeadModel` (`outputs.loss`).

### 3.7 Sinh nhạc
- **Điều khiển:** `temperature`, `top_k`, `top_p`.  
- **Giải mã token ID → triplet** (hàm TODO nếu chưa hoàn thành).

### 3.8 Đánh giá
- **Metric:** density, velocity, range, time-sig, tempo.  
- **Output:** `overall_score` ∈ [0,1].

---

## 4. File & thư mục được tạo theo thời gian
```text
[data/midi]             (input cố định)
└─ Step1 -> data/output/midi_metadata_list.json
└─ Step2 -> data/output/automated_paired_data.json
└─ Step3 -> data/output/text_embeddings.json
└─ Step4 -> data/output/clustered_text_data.json
└─ Step5 -> data/output/amt_training_data.json
└─ Step6 -> models/checkpoints/checkpoint_epoch_*.pt
└─ Step7 -> output/generated/*.mid
└─ Step8 -> data/evaluation/eval_report.json
```

---

## 5. Trình tự thực thi gợi ý (CLI)
```bash
cd AMT
# 1–5 trong một lệnh
python run.py pipeline

# 6 Huấn luyện
python run.py train

# 7 Sinh nhạc
python - <<'PY'
from AMT.source.model.generation import AMTGenerator
model = AMTGenerator("models/checkpoints/checkpoint_epoch_10.pt")
model.generate_music("A calm piano piece", "output/generated/calm_piano.mid")
PY

# 8 Đánh giá
python - <<'PY'
from AMT.source.evaluation.metrics import evaluate_generated_music
print(evaluate_generated_music("data/reference/ref.mid", "output/generated/calm_piano.mid"))
PY
```

---
**Tài liệu hoàn thành – Luồng dữ liệu được mô tả đầy đủ cho việc tái lập và debug.** 