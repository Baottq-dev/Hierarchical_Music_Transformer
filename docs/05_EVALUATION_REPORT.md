# 05_EVALUATION_REPORT.md

## 1. Mục tiêu
Đánh giá chất lượng nhạc sinh ra bằng nhiều metrics nâng cao, so sánh với reference, tổng hợp báo cáo.

## 2. Metrics sử dụng
- Note density, note range, velocity, tempo similarity
- BLEU, n-gram overlap, similarity
- Structural metrics (repetition, motif, ...)
- Benchmark tốc độ sinh nhạc

## 3. Logic tổng thể
- Tính toán metrics cho từng file sinh ra.
- So sánh với reference nếu có.
- Tổng hợp, xuất báo cáo, plot.

## 4. Class chính
- `EvaluationMetrics`: Tính toán metrics nâng cao.
- `ModelEvaluator`: Tổng hợp, so sánh, xuất báo cáo.

## 5. Input/Output
- **Input:** File MIDI sinh ra, file reference.
- **Output:**
  - `test_results/evaluation_report.json`
  - `test_results/detailed_metrics.json`
  - `test_results/comparisons.json`

## 6. Flow chi tiết
1. Tính metrics cho từng file.
2. So sánh với reference.
3. Tổng hợp, xuất báo cáo, plot.

## 7. Điểm mạnh
- Metrics đa dạng, đánh giá toàn diện.
- Hỗ trợ so sánh với reference.
- Plot, báo cáo chi tiết.

## 8. Hướng mở rộng
- Tích hợp thêm metrics mới (FAD, MUSHRA, ...).
- Đánh giá cảm xúc, thể loại, nhạc cụ tự động. 