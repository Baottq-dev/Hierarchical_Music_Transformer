# 07_PERFORMANCE_ANALYSIS_REPORT.md

## 1. Mục tiêu
Phân tích hiệu năng pipeline: tốc độ xử lý, huấn luyện, sinh nhạc, memory, benchmark.

## 2. Các chỉ số phân tích
- Thời gian xử lý MIDI/text (process)
- Thời gian huấn luyện/epoch, tổng training
- Thời gian sinh nhạc/sample, tổng test
- Memory usage từng bước
- Benchmark tốc độ, so sánh với baseline

## 3. Kết quả benchmark (ví dụ)
- Process: 1000 MIDI/text ~ 5 phút (CPU)
- Train: 10 epoch, batch 32, d_model 512 ~ 2h (GPU)
- Test: 1 sample ~ 2s (GPU), 10 sample ~ 25s (bao gồm đánh giá)

## 4. Điểm mạnh
- Tối ưu hóa batch, memory, tốc độ xử lý.
- Có thể chạy trên CPU/GPU, tự động nhận device.
- Log, thống kê chi tiết từng bước.

## 5. Hướng tối ưu
- Distributed training, multi-GPU.
- Tối ưu hóa memory (gradient checkpointing, mixed precision).
- Tối ưu hóa tốc độ sinh nhạc (fast sampling, parallel generation). 