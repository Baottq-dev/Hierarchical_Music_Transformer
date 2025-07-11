# Hướng dẫn chạy AMT (Automated Music Transcription)

Tài liệu này cung cấp hướng dẫn chi tiết để cài đặt và chạy dự án AMT.

## 1. Cài đặt

### 1.1 Yêu cầu hệ thống
- Python 3.9 trở lên
- pip
- Git
- 8GB RAM trở lên
- GPU có CUDA (khuyến nghị cho quá trình huấn luyện)

### 1.2 Cài đặt dependencies

```bash
# Clone repository (nếu chưa có)
# git clone https://github.com/username/AMT.git
# cd AMT

# Cài đặt package và dependencies
pip install -e .

# Cài đặt dependencies cho phát triển (nếu cần)
pip install -e ".[dev]"

# Cài đặt mô hình ngôn ngữ spaCy (cần cho text processing)
python -m spacy download en_core_web_sm
```

### 1.3 Cấu trúc thư mục
Đảm bảo có các thư mục sau (sẽ tự động tạo nếu chưa có):
```
AMT/
└── data/
    ├── midi/             # Thư mục chứa file MIDI gốc
    ├── reference/        # Thư mục MIDI tham chiếu đã được lọc
    ├── text/             # Chứa mô tả văn bản
    ├── processed/        # Dữ liệu đã xử lý
    ├── output/           # Kết quả đầu ra
    └── evaluation/       # Dữ liệu đánh giá
```

## 2. Quy trình chạy

AMT hoạt động theo một quy trình gồm nhiều bước. Dưới đây là hướng dẫn chi tiết cho từng bước:

### 2.1 Thu thập dữ liệu (Collect)

Bước này thu thập metadata từ file MIDI, tạo/thu thập mô tả văn bản, và ghép cặp chúng:

```bash
python collect.py --midi_dir data/midi --output_dir data/output --filter_quality
```

#### Các tùy chọn:
- `--midi_dir`: Thư mục chứa file MIDI (mặc định: data/reference)
- `--output_dir`: Thư mục đầu ra (mặc định: data/output)
- `--filter_quality`: Lọc dữ liệu theo chất lượng
- `--min_text_length`: Độ dài tối thiểu của văn bản (mặc định: 20)
- `--min_duration`: Thời lượng tối thiểu của file MIDI (mặc định: 10.0 giây)
- `--log_level`: Mức độ log (debug, info, warning, error, critical)

#### Pipeline thu thập dữ liệu:
1. Thu thập metadata từ file MIDI
2. Thu thập mô tả văn bản cho từng file MIDI
3. Tạo bộ dữ liệu hoàn chỉnh
4. Lọc dữ liệu theo chất lượng (nếu sử dụng `--filter_quality`)
5. Xác thực và thống kê bộ dữ liệu

#### Đầu ra:
- `midi_metadata.json`: Metadata chi tiết của các file MIDI
- `paired_data.json`: Dữ liệu ghép cặp giữa MIDI và văn bản
- `complete_dataset.json`: Tập dữ liệu hoàn chỉnh không lọc
- `complete_dataset_filtered.json`: (nếu sử dụng --filter_quality)

#### Lưu ý:
- Các lỗi "data byte must be in range 0..127" là bình thường khi xử lý các file MIDI không chuẩn
- Các file không hợp lệ sẽ được bỏ qua và quá trình xử lý vẫn tiếp tục

### 2.2 Xử lý dữ liệu (Process)

Bước này xử lý dữ liệu thô thành định dạng phù hợp cho huấn luyện, sử dụng lớp `AdvancedProcessor`:

#### 2.2.1 Xử lý một file đơn lẻ:
```bash
python process.py single path/to/file.mid
```

#### 2.2.2 Xử lý hàng loạt:
```bash
python process.py batch path/to/midi/dir --text-dir path/to/text/dir
```

#### 2.2.3 Tiếp tục từ checkpoint:
```bash
python process.py continue path/to/checkpoint.json
```

#### Các tùy chọn chính:
- `--mode`: Chế độ xử lý (single, batch, continue)
- `--use_hierarchical_encoding`: Sử dụng mã hóa phân cấp (khuyến nghị bật)
- `--use_relative_attention`: Sử dụng attention vị trí tương đối
- `--use_contextual_embeddings`: Sử dụng contextual embeddings
- `--max_sequence_length`: Độ dài tối đa của chuỗi (mặc định: 1024)
- `--batch_size`: Kích thước batch (mặc định: 32)
- `--device`: Thiết bị sử dụng (cuda, cpu)
- `--checkpoint_interval`: Số lượng file xử lý giữa các checkpoint
- `--num_workers`: Số lượng worker cho xử lý song song

### 2.3 Xử lý batched (cho bộ dữ liệu lớn)

Script tối ưu hóa cho việc xử lý các tập dữ liệu lớn, hỗ trợ xử lý song song:

```bash
python process_batched.py --input_dir data/reference --output_dir data/processed --workers 4 --use_gpu
```

#### Các tùy chọn:
- `--input_dir`: Thư mục đầu vào
- `--output_dir`: Thư mục đầu ra
- `--workers`: Số lượng worker cho xử lý song song
- `--batch_size`: Kích thước batch
- `--use_gpu`: Sử dụng GPU nếu có
- `--resume`: Tiếp tục xử lý từ lần trước
- `--hierarchical`: Sử dụng mã hóa phân cấp

### 2.4 Tiếp tục từ checkpoint

Nếu xử lý bị gián đoạn, bạn có thể tiếp tục từ checkpoint:

```bash
python continue_from_checkpoint.py --midi_checkpoint data/processed/midi_checkpoint.json --text_checkpoint data/processed/text_checkpoint.json --input_file data/output/paired_data.json --output_dir data/processed --workers 4 --use_gpu
```

#### Các tùy chọn quan trọng:
- `--midi_checkpoint`: Path đến file checkpoint MIDI
- `--text_checkpoint`: Path đến file checkpoint văn bản
- `--input_file`: File dữ liệu ghép cặp đầu vào
- `--output_dir`: Thư mục đầu ra cho dữ liệu đã xử lý
- `--workers`: Số lượng worker (mặc định: 4)
- `--use_gpu`: Sử dụng GPU cho xử lý văn bản
- `--use_cache`: Sử dụng cache để tăng tốc xử lý
- `--log_level`: Mức độ log
- `--checkpoint_interval`: Số batch giữa các lần lưu checkpoint
- `--batch_size`: Kích thước batch
- `--force_restart`: Bắt đầu lại từ đầu
- `--save_partial`: Lưu kết quả một phần nếu xử lý không hoàn thành

### 2.5 Tạo dữ liệu huấn luyện

Script này chuẩn bị dữ liệu cho huấn luyện mô hình, bao gồm phân chia tập dữ liệu và tăng cường dữ liệu:

```bash
python create_training_data.py --paired-data-file data/output/complete_dataset_filtered.json --output_dir data/processed --dataset-name advanced_dataset --split 0.8 0.1 0.1 --augment
```

#### Các tùy chọn:
- `--paired-data-file`: File dữ liệu ghép cặp đầu vào
- `--output_dir`: Thư mục đầu ra
- `--dataset-name`: Tên cho bộ dữ liệu
- `--max-sequence-length`: Độ dài tối đa của chuỗi token
- `--max-text-length`: Độ dài tối đa của văn bản
- `--batch-size`: Kích thước batch
- `--use-hierarchical-encoding`: Sử dụng mã hóa phân cấp
- `--use-contextual-embeddings`: Sử dụng contextual embeddings
- `--use-sentencepiece`: Sử dụng SentencePiece tokenizer
- `--augment`: Thực hiện augmentation dữ liệu
- `--device`: Thiết bị sử dụng (cuda, cpu)
- `--workers`: Số lượng worker

### 2.6 Huấn luyện mô hình

Huấn luyện Hierarchical Music Transformer với lớp `AdvancedTrainer`:

```bash
python train.py --paired-data-file data/processed/train_data.json --val-data-file data/processed/val_data.json --model-dir models/checkpoints --epochs 50 --batch-size 32 --lr 0.0001 --use-hierarchical-encoding
```

#### Các tùy chọn chính:
- `--paired-data-file`: File dữ liệu train đã ghép cặp
- `--val-data-file`: File dữ liệu validation
- `--model-dir`: Thư mục lưu mô hình
- `--epochs`: Số epoch
- `--batch-size`: Kích thước batch
- `--lr`: Learning rate
- `--optimizer`: Loại optimizer (adam, adamw, sgd)
- `--scheduler`: Loại scheduler (cosine, linear, step, none)
- `--use-hierarchical-encoding`: Sử dụng mã hóa phân cấp
- `--use-relative-attention`: Sử dụng attention vị trí tương đối
- `--max-grad-norm`: Ngưỡng gradient clipping
- `--early-stopping`: Số epoch không cải thiện trước khi dừng
- `--checkpoint-interval`: Số epoch giữa các checkpoint
- `--generate-samples`: Sinh mẫu trong quá trình validation
- `--device`: Thiết bị sử dụng (cuda, cpu)

### 2.7 Sinh nhạc từ văn bản

Sử dụng mô hình đã huấn luyện để tạo MIDI từ mô tả văn bản:

```bash
python generate.py text --model models/checkpoints/best_model.pt --text "A cheerful piano melody with jazz influences"
```

#### Các chế độ:
- `text`: Sinh MIDI từ văn bản
- `file`: Sinh MIDI từ file văn bản
- `midi`: Sinh văn bản từ MIDI
- `batch`: Xử lý hàng loạt

#### Các tùy chọn:
- `--model`: Đường dẫn đến mô hình đã huấn luyện
- `--text`: Mô tả văn bản (cho chế độ text)
- `--input_file`: File đầu vào (cho chế độ file/batch)
- `--output_dir`: Thư mục đầu ra
- `--model_type`: Loại mô hình (transformer, bidirectional)
- `--temperature`: Nhiệt độ cho quá trình sinh (mặc định: 0.7)
  - Thấp (0.3-0.5): Kết quả ổn định, ít sáng tạo
  - Cao (0.7-1.0): Kết quả đa dạng, sáng tạo hơn
- `--top_k`: Tham số top-k (mặc định: 50)
- `--top_p`: Tham số top-p (mặc định: 0.95)
- `--repetition_penalty`: Phạt lặp lại (mặc định: 1.0)
- `--device`: Thiết bị sử dụng (cuda, cpu)

### 2.8 Kiểm thử và đánh giá

Đánh giá hiệu suất của mô hình với nhiều chế độ kiểm thử:

```bash
python test.py text_to_midi --model models/checkpoints/best_model.pt --input_file data/evaluation/texts.txt
```

#### Các chế độ:
- `text_to_midi`: Kiểm thử chuyển từ văn bản sang MIDI
- `midi_to_text`: Kiểm thử chuyển từ MIDI sang văn bản
- `parameter_sweep`: Thử nghiệm với nhiều tham số khác nhau
- `test_data`: Đánh giá trên tập test

#### Các tùy chọn:
- `--model`: Đường dẫn đến mô hình
- `--input_file`/`--input_dir`: File/thư mục đầu vào
- `--reference_dir`/`--reference_file`: Dữ liệu tham chiếu
- `--output_dir`: Thư mục đầu ra
- `--model_type`: Loại mô hình (transformer, bidirectional)
- `--metrics`: Các metrics muốn tính toán
- `--temperature`, `--top_k`, `--top_p`: Tham số sinh
- `--device`: Thiết bị sử dụng (cuda, cpu)

## 3. Ví dụ quy trình đầy đủ

Dưới đây là một quy trình đầy đủ từ thu thập dữ liệu đến tạo nhạc:

```bash
# 1. Cài đặt
pip install -e .
python -m spacy download en_core_web_sm

# 2. Thu thập dữ liệu
python collect.py --midi_dir data/midi --output_dir data/output --filter_quality --min_text_length 30 --min_duration 15.0

# 3. Xử lý dữ liệu
python process_batched.py --input_dir data/output --output_dir data/processed --workers 4 --use_gpu --hierarchical

# 4. Tạo dữ liệu huấn luyện
python create_training_data.py --paired-data-file data/output/complete_dataset_filtered.json --output_dir data/processed --dataset-name music_dataset --use-hierarchical-encoding --augment

# 5. Huấn luyện mô hình
python train.py --paired-data-file data/processed/music_dataset_train.json --val-data-file data/processed/music_dataset_val.json --model-dir models/checkpoints --epochs 50 --batch-size 32 --use-hierarchical-encoding --early-stopping 5

# 6. Sinh nhạc từ văn bản
python generate.py text --model models/checkpoints/best_model.pt --text "A beautiful piano melody with a melancholic mood" --temperature 0.8 --top_k 50 --top_p 0.95

# 7. Đánh giá
python test.py text_to_midi --model models/checkpoints/best_model.pt --input_file data/evaluation/texts.txt
```

## 4. Xử lý lỗi và gỡ rối

### 4.1 Xử lý file MIDI không hợp lệ
Khi gặp lỗi "data byte must be in range 0..127":
```bash
# Tạo một thư mục midi_filtered
mkdir -p data/midi_filtered
# Chạy collect với log_level = debug để xem chi tiết các file bị lỗi
python collect.py --midi_dir data/midi --output_dir data/output --log_level debug
```

### 4.2 Checkpoint không hoạt động
Nếu tiếp tục từ checkpoint không hoạt động:
```bash
python process.py continue path/to/checkpoint.json --force_restart
```
hoặc
```bash
python continue_from_checkpoint.py --input_file data/output/paired_data.json --output_dir data/processed --force_restart
```

### 4.3 Lỗi CUDA out of memory
Giảm kích thước batch và/hoặc độ dài chuỗi:
```bash
python train.py --batch-size 16 --max-sequence-length 512 [other options]
```

### 4.4 Theo dõi quá trình huấn luyện
Sử dụng TensorBoard:
```bash
tensorboard --logdir logs/
```

### 4.5 Lỗi import
Nếu gặp lỗi "cannot import name X", kiểm tra:
```bash
# Đảm bảo đã cài đặt package trong mode development
pip install -e .

# Kiểm tra file __init__.py trong module tương ứng
cat amt/module/__init__.py
```

### 4.6 Kiểm thử nhanh
Chạy kiểm thử nhanh để xác nhận cài đặt:
```bash
pytest tests/integration/test_fast.py
```

## 5. Mẹo và gợi ý

1. **Dữ liệu**: Chất lượng dữ liệu ảnh hưởng lớn đến hiệu suất:
   - Sử dụng `--filter_quality` với giá trị hợp lý (min_text_length ≥ 20, min_duration ≥ 10.0)
   - Xem xét tăng ngưỡng nếu muốn dữ liệu chất lượng cao hơn

2. **GPU**: Sử dụng GPU cho quá trình huấn luyện và xử lý văn bản:
   - Dùng flag `--use_gpu` hoặc `--device cuda` khi khả dụng
   - CUDA out of memory: Giảm batch size và/hoặc độ dài chuỗi

3. **Checkpoint**: Đặt `--checkpoint_interval` nhỏ hơn nếu:
   - Dữ liệu quý giá hoặc không ổn định
   - Môi trường tính toán không ổn định
   - Giá trị 10-50 thường hợp lý tùy kích thước dữ liệu

4. **Augmentation**: Sử dụng augmentation dữ liệu khi:
   - Số lượng mẫu huấn luyện ít (< 1000 mẫu)
   - Cần đa dạng hóa dữ liệu
   - Muốn cải thiện khả năng tổng quát hóa

5. **Tham số sinh**: Điều chỉnh tùy theo nhu cầu:
   - Temperature: 
     - Thấp (0.3-0.5): Ổn định, ít sáng tạo, tốt cho sản phẩm
     - Cao (0.7-1.0): Đa dạng, sáng tạo, thử nghiệm nhiều hơn
   - Top-k (30-100): Giá trị cao = nhiều lựa chọn hơn = sáng tạo hơn
   - Top-p (0.85-0.97): Giá trị cao = đa dạng hơn

6. **Mã hóa phân cấp**: Luôn bật `--use-hierarchical-encoding` cho kết quả tốt hơn:
   - Cải thiện cấu trúc âm nhạc
   - Nắm bắt tốt hơn các mối quan hệ nhạc lý

7. **Xử lý song song**: Tối ưu số workers theo cấu hình máy:
   - CPU: Đặt `--workers` bằng số core - 1
   - GPU: 2-4 workers đủ để giữ GPU hoạt động liên tục

8. **Lưu ý về dữ liệu MIDI**: 
   - Một số file MIDI có thể không hợp lệ hoặc không tuân theo chuẩn
   - Các lỗi "data byte must be in range 0..127" là bình thường
   - Hệ thống sẽ bỏ qua các file không hợp lệ và tiếp tục xử lý 