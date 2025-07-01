# 04_MUSIC_GENERATION_REPORT.md

## 1. Mục tiêu
Sinh nhạc từ text, đánh giá chất lượng nhạc sinh ra, so sánh với reference, benchmark hiệu năng.

## 2. Logic tổng thể
- Load mô hình đã huấn luyện.
- Sinh nhạc từ text đầu vào hoặc batch.
- Đánh giá nhạc sinh ra bằng nhiều metrics (note density, velocity, BLEU, similarity, ...).
- So sánh với reference, benchmark tốc độ, xuất báo cáo.

## 3. Class chính
- `ModelTester`: Sinh nhạc, kiểm thử mô hình.
- `ModelEvaluator`: Đánh giá nhạc sinh ra, so sánh với reference.
- `EvaluationMetrics`: Tính toán các metrics nâng cao.

## 4. Input/Output
- **Input:**
  - Mô hình đã huấn luyện (`models/checkpoints/best_model.pt`)
  - File MIDI sinh ra (`data/output/generated*.mid`)
  - File reference (`data/midi/ref*.mid`)
- **Output:**
  - `test_results/evaluation_report.json`
  - `test_results/detailed_metrics.json`
  - `test_results/comparisons.json`
  - Các plot, log, file sinh ra khác

## 5. Flow chi tiết
1. Load mô hình, kiểm thử loading.
2. Sinh nhạc từ text hoặc batch file.
3. Đánh giá từng file sinh ra, tính metrics.
4. So sánh với reference, benchmark tốc độ.
5. Xuất báo cáo, plot, log.

## 6. Điểm mạnh
- Đánh giá đa chiều, nhiều metrics nâng cao.
- So sánh trực tiếp với reference.
- Benchmark tốc độ sinh nhạc.

## 7. Hướng mở rộng
- Tích hợp thêm metrics mới (FAD, MUSHRA, ...).
- Hỗ trợ sinh nhạc đa modal, đa nhạc cụ.
- Tối ưu hóa tốc độ, giao diện đánh giá. 