# Chi tiết cấu trúc dự án AMT (Automated Music Transcription)

Dự án AMT là hệ thống phiên âm nhạc tự động sử dụng mô hình Transformer để chuyển đổi giữa biểu diễn âm nhạc tượng trưng (MIDI) và văn bản mô tả.

## Cấu trúc thư mục chính

### 1. `amt/` - Thư mục chính chứa mã nguồn

#### 1.1. `__init__.py`
- Định nghĩa module gốc của package
- Import các thành phần chính để làm chúng có sẵn ở cấp gói

#### 1.2. `cli_simple.py` & `cli.py`
- `cli_simple.py`: Giao diện dòng lệnh đơn giản
- `cli.py`: Giao diện dòng lệnh đầy đủ với nhiều tính năng hơn
- Cả hai đều cung cấp giao diện dòng lệnh cho các chức năng chính của dự án

#### 1.3. `config.py`
- Quản lý cấu hình cho toàn bộ dự án
- Sử dụng Pydantic để xác thực cài đặt
- Đọc cấu hình từ biến môi trường hoặc file cấu hình

#### 1.4. `collect/` - Module thu thập dữ liệu
- `__init__.py`: Định nghĩa module và export các lớp chính
- `collector.py`: Lớp cơ sở cho việc thu thập dữ liệu
- `data_pairing.py`: Ghép cặp dữ liệu MIDI với mô tả văn bản
  - Cung cấp chức năng lọc dữ liệu theo chất lượng với các tiêu chí:
    - Độ dài tối thiểu của văn bản mô tả (mặc định: 20 ký tự)
    - Thời lượng tối thiểu của file MIDI (mặc định: 10.0 giây)
- `midi_collector.py`: Thu thập và xử lý file MIDI
  - Trích xuất metadata từ các file MIDI như: tempo, thời lượng, nhạc cụ, v.v.
  - Xử lý lỗi khi gặp file MIDI không hợp lệ
- `text_collector.py`: Thu thập mô tả văn bản cho file MIDI
  - Tìm kiếm thông tin trên Wikipedia
  - Tự động tạo mô tả từ tên file khi không có thông tin từ Wikipedia

#### 1.5. `process/` - Module xử lý dữ liệu
- `__init__.py`: Định nghĩa module
- `continue_from_checkpoint.py`: Tiếp tục xử lý từ điểm dừng trước đó
  - Hỗ trợ xử lý tập dữ liệu lớn với khả năng tạm dừng và tiếp tục
- `data_preparer.py`: Chuẩn bị dữ liệu cho quá trình huấn luyện
  - Tokenize và chuyển đổi định dạng dữ liệu
  - Tạo các dataloader cho PyTorch
- `midi_processor.py`: Xử lý file MIDI thành token
  - Phân tích và mã hóa cấu trúc âm nhạc
  - Biểu diễn phân cấp với các token ở cấp bar, beat, và note
- `text_processor.py`: Xử lý văn bản thành vector nhúng
  - Sử dụng SentencePiece và BERT để tạo embedding

#### 1.6. `train/` - Module huấn luyện
- `__init__.py`: Định nghĩa module và export các lớp/hàm chính
- `create_training_data.py`: Tạo dữ liệu huấn luyện từ dữ liệu đã xử lý
  - Chứa lớp `AdvancedDataCreator` và hàm `create_advanced_training_data`
  - Thực hiện tăng cường dữ liệu qua các kỹ thuật chuyển đổi
- `model.py`: Định nghĩa kiến trúc mô hình
  - Triển khai lớp `MusicTransformer`
- `trainer.py`: Quản lý quá trình huấn luyện
  - Cung cấp lớp `ModelTrainer`
  - Theo dõi và ghi lại các metrics huấn luyện
- `training_loop.py`: Vòng lặp huấn luyện mô hình
  - Thực hiện các epochs huấn luyện
  - Lưu các checkpoint

#### 1.7. `evaluate/` - Module đánh giá
- `__init__.py`: Định nghĩa module
- `evaluator.py`: Đánh giá hiệu suất của mô hình
  - Cung cấp lớp `ModelEvaluator`
  - Tính toán các chỉ số đánh giá
- `metrics.py`: Các metrics để đánh giá chất lượng
  - BLEU, ROUGE cho mô tả văn bản
  - Các metrics âm nhạc đặc biệt cho MIDI
- `tester.py`: Kiểm thử mô hình trên dữ liệu mới
  - Cung cấp lớp `ModelTester`
  - Chạy các bài test để đánh giá khả năng tổng quát hóa

#### 1.8. `generate/` - Module tạo nội dung
- `__init__.py`: Định nghĩa module
- `generator.py`: Tạo MIDI từ mô tả văn bản hoặc ngược lại
  - Cung cấp lớp `MusicGenerator`
  - Áp dụng các kỹ thuật sampling (top-k, top-p)
  - Kiểm soát nhiệt độ (temperature) trong quá trình sinh

#### 1.9. `models/` - Định nghĩa các mô hình
- `hierarchical_music_transformer.py`: Mô hình Transformer phân cấp cho âm nhạc
  - Cung cấp lớp `HierarchicalMusicTransformer`
  - Hàm `create_transformer_model` để tạo mô hình với các tham số tùy chọn
  - Mã hóa các thành phần âm nhạc theo cấu trúc phân cấp
  - Xử lý các mối quan hệ phức tạp giữa các thành phần âm nhạc

#### 1.10. `utils/` - Các tiện ích
- `__init__.py`: Định nghĩa module
- `logging.py`: Cấu hình và quản lý log
  - Cung cấp hàm `get_logger` để tạo logger với định dạng nhất quán

### 2. Các file gốc

#### 2.1. `collect.py`
- Script để thu thập dữ liệu MIDI và văn bản
- Tương tác với `amt.collect` để tạo bộ dữ liệu ghép cặp
- Pipeline:
  1. Thu thập metadata từ file MIDI 
  2. Thu thập mô tả văn bản cho từng file MIDI
  3. Tạo bộ dữ liệu hoàn chỉnh
  4. Lọc dữ liệu theo chất lượng (tùy chọn)
  5. Xác thực và thống kê bộ dữ liệu
- Tham số chính:
  - `--midi_dir`: Thư mục chứa các file MIDI
  - `--output_dir`: Thư mục đầu ra
  - `--filter_quality`: Lọc dữ liệu theo chất lượng
  - `--min_text_length`: Độ dài tối thiểu của văn bản
  - `--min_duration`: Thời lượng tối thiểu của file MIDI

#### 2.2. `process.py`
- Script xử lý dữ liệu MIDI và văn bản
- Sử dụng lớp `AdvancedProcessor` 
- Chuyển đổi dữ liệu thô thành dạng có thể sử dụng cho huấn luyện
- Hỗ trợ các chế độ:
  - `single`: Xử lý một file đơn lẻ
  - `batch`: Xử lý hàng loạt file
  - `continue`: Tiếp tục từ checkpoint

#### 2.3. `train.py`
- Script huấn luyện mô hình
- Sử dụng lớp `AdvancedTrainer`
- Cấu hình và theo dõi quá trình huấn luyện
- Lưu các checkpoint và log metrics
- Hỗ trợ nhiều tùy chọn mô hình và tối ưu hóa

#### 2.4. `generate.py`
- Script tạo nội dung từ mô hình đã huấn luyện
- Hỗ trợ nhiều chế độ: text, file, midi, batch
- Tạo MIDI từ văn bản hoặc văn bản từ MIDI
- Điều chỉnh các tham số sinh như temperature, top-k, top-p

#### 2.5. `test.py`
- Script kiểm thử và đánh giá mô hình
- Tính toán metrics hiệu suất
- Hỗ trợ các chế độ kiểm thử khác nhau:
  - `text_to_midi`: Kiểm thử chuyển từ văn bản sang MIDI
  - `midi_to_text`: Kiểm thử chuyển từ MIDI sang văn bản
  - `parameter_sweep`: Thử nghiệm với nhiều tham số khác nhau
  - `test_data`: Đánh giá trên tập test

#### 2.6. `continue_from_checkpoint.py`
- Script để tiếp tục xử lý từ một checkpoint
- Gọi hàm `continue_from_checkpoint` từ module `amt.process`
- Hữu ích cho việc xử lý các tập dữ liệu lớn
- Khôi phục trạng thái và tiếp tục từ điểm dừng

#### 2.7. `create_training_data.py`
- Script tạo dữ liệu huấn luyện từ dữ liệu đã xử lý
- Gọi hàm `create_advanced_training_data` từ module `amt.train`
- Thực hiện các biến đổi và chuẩn bị cho việc huấn luyện

#### 2.8. `process_batched.py`
- Script xử lý dữ liệu theo batch
- Tối ưu hóa cho các tập dữ liệu lớn
- Hỗ trợ xử lý song song thông qua tham số `--workers`

### 3. Thư mục dữ liệu

#### 3.1. `data/`
- `evaluation/`: Dữ liệu dùng cho đánh giá mô hình
- `output/`: Kết quả đầu ra từ quá trình xử lý
  - `midi_metadata.json`: Metadata của các file MIDI
  - `paired_data.json`: Dữ liệu ghép cặp giữa MIDI và văn bản
  - `complete_dataset.json`: Bộ dữ liệu hoàn chỉnh không lọc
  - `complete_dataset_filtered.json`: Bộ dữ liệu đã lọc theo chất lượng
- `processed/`: Dữ liệu đã qua xử lý
  - Các checkpoint trong quá trình xử lý
  - Dữ liệu đã tokenize và chuẩn bị cho huấn luyện
- `reference/`: Dữ liệu gốc (file MIDI, âm thanh)
- `text/`: Các mô tả văn bản

### 4. Tài liệu và hướng dẫn

#### 4.1. `docs/`
- `04_method.md`: Phương pháp tiếp cận trong dự án
- `05_model_deep_dive.md`: Chi tiết về kiến trúc mô hình
- `checkpoint_system.md`: Giải thích về hệ thống checkpoint
- `configuration.md`: Hướng dẫn cấu hình
- `docs/`: Tài liệu API và hướng dẫn sử dụng
  - `api/`: Tài liệu tham khảo API
  - `usage/`: Hướng dẫn sử dụng
  - `overview/`: Tổng quan về kiến trúc
- `README.md`: Giới thiệu tổng quan về tài liệu

### 5. Kiểm thử

#### 5.1. `tests/`
- `__init__.py`: Định nghĩa package test
- `conftest.py`: Cấu hình pytest
- `fixtures/`: Dữ liệu kiểm thử
- `integration/`: Kiểm thử tích hợp
  - `test_fast.py`: Các bài kiểm thử nhanh
  - `test_processing_pipeline.py`: Kiểm thử quy trình xử lý
- `unit/`: Kiểm thử đơn vị
  - `test_cli.py`: Kiểm thử giao diện dòng lệnh
  - `test_config.py`: Kiểm thử hệ thống cấu hình
  - `test_midi_processor.py`: Kiểm thử xử lý MIDI
  - `test_settings.py`: Kiểm thử cài đặt
  - `test_simple.py`: Các kiểm thử đơn giản khác

### 6. Cấu hình dự án

#### 6.1. `pyproject.toml`
- Cấu hình project theo PEP 621
- Khai báo các dependencies chính:
  - numpy, torch, pretty_midi, mido, tqdm
  - tensorboard, click, pydantic
- Khai báo dependencies phát triển (black, isort, ruff, mypy, pytest)
- Cấu hình các công cụ phát triển:
  - black: Định dạng mã nguồn
  - isort: Sắp xếp import
  - ruff: Linter
  - mypy: Kiểm tra kiểu tĩnh

#### 6.2. `requirements.txt`
- Danh sách các gói phụ thuộc
- Chia thành các nhóm:
  - Core dependencies (numpy, torch, pretty_midi...)
  - Xử lý MIDI (music21)
  - Xử lý văn bản (transformers, sentencepiece, spacy)
  - Tùy chọn (tensorboard)
  - Phát triển (black, isort, ruff, pytest...)

#### 6.3. `setup.cfg`
- Cấu hình setuptools
- Khai báo entry points cho CLI:
  - amt = "amt.cli:cli"
  - amt-simple = "amt.cli_simple:cli"

## Luồng làm việc điển hình

1. Thu thập dữ liệu (`collect.py`)
   - Thu thập metadata MIDI
   - Thu thập mô tả văn bản
   - Ghép cặp dữ liệu
   - Lọc theo chất lượng (tùy chọn)
   
2. Xử lý dữ liệu (`process.py` hoặc `process_batched.py`)
   - Chuyển đổi dữ liệu thô thành tokens
   - Mã hóa phân cấp cho MIDI
   - Tạo embeddings cho văn bản
   
3. Tạo dữ liệu huấn luyện (`create_training_data.py`)
   - Chia tập train/val/test
   - Augment dữ liệu (tùy chọn)
   - Tạo datasets và dataloaders
   
4. Huấn luyện mô hình (`train.py`)
   - Huấn luyện Hierarchical Music Transformer
   - Theo dõi metrics qua TensorBoard
   - Lưu checkpoint tốt nhất
   
5. Tạo nội dung mới (`generate.py`)
   - Tạo MIDI từ mô tả văn bản
   - Điều chỉnh tham số sinh (temperature, top-k, top-p)
   
6. Đánh giá kết quả (`test.py`)
   - Tính toán metrics
   - Đánh giá hiệu suất

## Các tính năng đặc biệt

1. **Mã hóa phân cấp**: Biểu diễn âm nhạc theo cấu trúc phân cấp (bar, beat, note)
2. **Hệ thống checkpoint**: Khả năng tạm dừng và tiếp tục xử lý với các tập dữ liệu lớn
3. **Mã hóa vị trí tương đối**: Cải thiện việc mô hình hóa các mối quan hệ thời gian trong âm nhạc
4. **Tăng cường dữ liệu**: Tự động mở rộng dữ liệu huấn luyện thông qua các biến đổi âm nhạc
5. **Sinh có kiểm soát**: Điều chỉnh quá trình sinh qua temperature, top-k, top-p
6. **Xử lý lỗi**: Khả năng bỏ qua các file MIDI không hợp lệ và tiếp tục xử lý 