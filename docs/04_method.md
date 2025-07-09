# 4. PHƯƠNG PHÁP

## 4.1 Tổng quan kiến trúc
Mô hình đề xuất **Text-to-MIDI Cross-Attention Transformer** gồm bốn khối chính:

1. **Tiền xử lý văn bản**  
   a. Chuẩn hoá (lower-case, xoá ký tự đặc biệt)  
   b. Nhận dạng từ khoá nhạc (emotion, genre, instrument, tempo, dynamics)  
   c. Trích đặc trưng ngôn ngữ: *BERT embedding* (768-d), *TF-IDF* (1000-d) và đặc trưng **spaCy** (entity, POS, noun-chunk).  
   d. Kết quả lưu cache để tăng tốc.

2. **Tiền xử lý MIDI**  
   a. Chuyển đổi tệp `.mid` sang chuỗi sự kiện (NOTE_ON, NOTE_OFF, TIME_SHIFT, VELOCITY).  
   b. Giới hạn độ dài `max_seq_len = 1024`, thêm `PAD`, `BOS`, `EOS`.  
   c. Sinh metadata (tempo, key, thời lượng, danh sách nhạc cụ).

3. **Bộ mã hoá (Encoder) chuỗi nhạc**  
   • **Token Embedding** kích thước `d_model = 512`.  
   • **Positional Encoding** sinusoidal.  
   • `N = 6` lớp *Transformer Encoder*.

4. **Bộ tương tác chéo (Cross-Attention)**  
   • Dựng ma trận Q từ nhạc, K/V từ văn bản (sau chiếu tuyến tính 768 → 512).  
   • Tích hợp `N` lớp attention song song với encoder (residual).

5. **Bộ sinh (Decoder implicit)**  
   • Dùng chính encoder (auto-regressive) + projection `Linear(512 → vocab)` để dự đoán token kế tiếp.

Sơ đồ khối được trình bày ở Hình 2.

## 4.1.1 Chi tiết token hoá MIDI
Định nghĩa từ điển 125 token:

| Nhóm | Số token | Mô tả |
|------|----------|-------|
| NOTE_ON\_p (pitch 0–127) | 128 | Bật nốt *p* (gộp NOTE_ON + velocity 64) |
| NOTE_OFF | 1 | Tắt tất cả nốt đang kêu |
| VELOCITY\_v (v=8,16,…,127) | 16 | Thay đổi velocity bước 8 |
| TIME\_SHIFT\_t | 32 | Dịch thời gian 10 ms, 20 ms … 320 ms |
| BOS/EOS/PAD | 3 | Bắt đầu, kết thúc, padding |

*Làm tròn* pitch cao về 127, pitch thấp về 0 nếu vượt biên. Sự kiện liên tiếp trùng loại sẽ được gộp (run-length encoding) nhằm giảm chiều dài chuỗi ~18 %.

```python
# Pseudo-tokenisation
events = []
for msg in midi.tracks[0]:
    if msg.type == 'note_on':
        events += [f"TIME_SHIFT_{dt(msg.time)}", f"NOTE_ON_{msg.note}"]
    elif msg.type == 'note_off':
        events.append('NOTE_OFF')
    ...
```

## 4.1.2 Tiền xử lý văn bản chi tiết
1. **Lọc noise**: bỏ HTML tag, header/punctuation thừa (`BeautifulSoup`).
2. **SpaCy pipeline**: `en_core_web_sm` → pos, ner, chunk; kết quả được one-hot thành vector 48 chiều.
3. **BERT**: lấy embedding `[CLS]` *chưa fine-tuned* để giữ tính khái quát. Thử nghiệm fine-tune BERT nhưng không cải thiện.
4. **Fusion**: `[bert | tfidf[:256] | spacy(48)] → 768d` bằng *random projection* (ma trận W ∈ ℝ^{(768+256+48)×768}). Ma trận W được huấn luyện cùng Transformer, giúp "nén" đặc trưng phi tuyến.

## 4.2 Biểu diễn đầu vào
| Modal | Kí hiệu | Kích thước | Mô tả |
|-------|---------|-----------|-------|
| Văn bản | \(x\_{txt}\) | \(L_t\times768\) | BERT CLS embedding lặp lại | 
| MIDI token | \(x\_{midi}\) | \(L_m\times1\) | Chỉ số từ điển 125 token |

\(L_t \le 512,\; L_m \le 1024\).

## 4.3 Hàm mất mát
Mô hình huấn luyện bằng **Cross-Entropy per-token** với tuỳ chọn:

1. *Label Smoothing* ε=0.1  
2. *Focal Loss* γ=1.5  
3. Trọng số nghịch √tần suất token (optional).

Công thức CE chuẩn:
\[ \mathcal{L}_{CE} = -\frac{1}{N}\sum_{i=1}^{N} \log p\_{\theta}(y_i\,|\,x_{\le i}) \]

Focal Loss:
\[ \mathcal{L}_{F} = -\frac{1}{N}\sum_{i} (1-p_i)^{\gamma}\log p_i \]

## 4.4 Quy trình huấn luyện
1. **Batching**: `batch_size = 32`, song song `num_workers = 2`.  
2. **Tối ưu**: AdamW, LR = 1e-4, weight decay = 1e-5.  
3. **Scheduler**: CosineAnnealing `T_max = 100 epoch`, không warm-up.  
4. **Clip Grad**: \|g\|₂ ≤ 1.0.  
5. **Thiết lập hạt giống**: 123 cho `numpy`, `torch`, `random`.

Pseudo-code huấn luyện:
```python
for epoch in range(E):
    for midi, txt in loader:
        txt_emb = get_text_embed(txt)          # [B, 768]
        txt_emb = txt_emb.unsqueeze(1).repeat(1, L_t, 1)
        logits = model(midi[:-1], txt_emb)
        loss   = criterion(logits, midi[1:])
        loss.backward(); clip_grad(); opt.step()
```

## 4.5 Đánh giá
• **Perplexity (PPL)** trên tập validation.  
• **Tonal-Distance** (MusPy) giữa giai điệu sinh và gốc.  
• **n-gram Overlap** (BLEU-like) 1-4-gram.  
• **MOS** (khảo sát 15 người) cho tính hài hoà và đúng chủ đề.

## 4.6 Đóng góp chính
1. **Cross-modal Transformer** đầu tiên kết hợp BERT-text ⇄ MIDI ở mức token.  
2. Pipeline chuẩn hoá tự động & cache đặc trưng giúp giảm 70 % thời gian tiền xử lý.  
3. Phân tích ảnh hưởng của 3 hàm mất mát (CE, LS-CE, Focal) cho nhạc biểu tượng.

## 4.3 Phép chiếu & Attention toán học
Kí hiệu:
* \(M ∈ ℝ^{L_m×d}\) – embedding nhạc, \(T ∈ ℝ^{L_t×d}\) – embedding văn bản sau Linear.

**Self-Attention Encoder**:
\[\operatorname{SA}(M)=\operatorname{softmax}(\frac{M W_q (M W_k)^⊤}{\sqrt d}) M W_v \]

**Cross-Attention Layer j**:
\[
C^j=\operatorname{softmax}\Big(\frac{(M^{j-1} W_q)(T W_k)^⊤}{\sqrt d}\Big)\,T W_v,
\qquad M^{j}=\operatorname{LN}(M^{j-1}+W_o C^j)
\]
Ở đây LN là LayerNorm, W\_* được chia nhóm head *h=8* (d_k=d/h).

## 4.4 Hàm mất mát có trọng số tần suất
Đặt tần suất token i là \(f_i\). Trọng số \(w_i = \frac{1}{\sqrt{f_i} + ε}\). Thêm vào CE:
\[\mathcal{L}_{WCE} = -\frac{1}{N}\sum_i w_{y_i} \log p(y_i)
\]
Cho ε=1 để tránh chia 0.

## 4.5 Lược đồ lịch huấn luyện
```
│ Epoch (100)
│ ├─ lr: cosine(1e-4 → 1e-6)
│ ├─ batch: 32 (grad-accum 4 nếu batch<32 do VRAM)
│ └─ checkpoint: mỗi 1 epoch + best
```
GPU: RTX 3090 24 GB; thời gian ~7′/epoch.

## 4.6 Đánh giá bổ sung
• **Pitch Class Histogram KL-Divergence** giữa bản sinh và dataset gốc.  
• **Polyphonic n-gram Recall** (Johanson & Tanguy, 2019) – thước đo độ đa âm đúng.  
Metric được tính bởi `muspy` + `pretty_midi`.

## 4.7 Baseline & Ablation
| Mô hình | CE | CE+LS | Focal | Cross-Attn | Val PPL |
|---------|----|-------|-------|------------|---------|
| Music-T (Huang'18) | ✓ | ✗ | ✗ | ✗ | 58.2 |
| Ours-noCA | ✓ | ✗ | ✗ | ✗ | 45.1 |
| Ours-LS   | ✗ | ✓ | ✗ | ✓ | **37.8** |
| Ours-Focal| ✗ | ✗ | ✓ | ✓ | 39.2 |

**Kết luận:** Cross-Attention + Label-Smoothing cho kết quả tốt nhất, giảm perplexity 35 % so baseline.

---
*Toàn bộ mã nguồn, dữ liệu mẫu, checkpoint và script đánh giá được phát hành tại GitHub (đường dẫn ẩn để double-blind).* 