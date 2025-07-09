# 5. PHÂN TÍCH SÂU KIẾN TRÚC MÔ HÌNH

*Phiên bản làm lại – Markdown thuần, không ký tự đặc biệt gây lỗi render.*

---
## 5.1 Kiến trúc tầng cao

```
              ┌──────────────────────────────┐
              │  Text Processor              │
              │  (clean + BERT + TF-IDF)     │
Input Text ─▶ │  → T ∈ R^{L_t×768}           │
              └───────────────┬──────────────┘
                              │  (Linear 768→512 + PE)
                              ▼
              ┌──────────────────────────────┐
              │  Music Transformer Encoder   │
MIDI tokens ─▶│  6 × [SA → CA → FF]          │
              │  d_model = 512, h = 8        │
              └───────────────┬──────────────┘
                              │  (Self-Attn output)
                              ▼
              ┌──────────────────────────────┐
              │   Linear 512→vocab           │
              └──────────────────────────────┘
```
* SA: Self-Attention trên chuỗi nhạc.  
* CA: Cross-Attention (Q = nhạc, K/V = text).

---
## 5.2 Embedding & Positional Encoding

* **Token embedding** \(E_m ∈ R^{V×d}\) với `V = 125`, `d = 512`.
* **Positional encoding** sinusoidal cho nhạc (\(L_m ≤ 1024\)) và text (\(L_t ≤ 512\)).

\[
M^0 = E_m[y] + P^{(m)}, \qquad T = W_p X + P^{(t)}
\]
`W_p` là ma trận chiếu 768→512.

---
## 5.3 Khối Transformer một lớp

```
# input x = M^{j-1}
1) x   = LN( x + SA(x) )
2) if cross: x = LN( x + CA(x, T) )
3) x   = LN( x + FF(x) )
```
* Self-Attention: multi-head (`h = 8`, `d_k = 64`).  
* Cross-Attention: Query = x, Key/Value = T.

Công thức CA:
\[
Q = x W_q, \; K = T W_k, \; V = T W_v \newline
S = \frac{QK^\top}{\sqrt d}, \; C = \text{softmax}(S) V, \; \text{CA}(x,T)= W_o C
\]

---
## 5.4 Token hoá MIDI (125 token)

| Loại            | Số lượng | Mô tả                             |
|-----------------|----------|-----------------------------------|
| NOTE_ON_p       | 128      | pitch 0-127 (velocity mặc định)   |
| NOTE_OFF        | 1        | tắt toàn bộ nốt đang kêu          |
| VELOCITY_v      | 16       | v = 8,16,…,127                    |
| TIME_SHIFT_t    | 32       | t = 10-320 ms (bước 10 ms)        |
| BOS / EOS / PAD | 3        | đặc biệt                          |

Sự kiện liên tiếp cùng loại được gộp (run-length) để giảm chuỗi ~18 %.

---
## 5.5 Hàm mất mát

1. **Cross-Entropy** (mặc định)  
2. **Label-Smoothing**: \(\varepsilon = 0.1\)  
3. **Focal Loss**: \(\gamma = 1.5\)  
4. **Weighted CE**: trọng số \(w_i = 1/\sqrt{f_i + 1}\).

---
## 5.6 Pseudo-code forward

```python
# midi: [B, L_m]; text_emb: [B, L_t, 768]
text_proj = proj(text_emb) + pe_text            # [B, L_t, 512]
mid_emb   = emb(midi) + pe_midi                 # [B, L_m, 512]

x = mid_emb
for block in transformer_layers:
    x = block(x, text_proj)  # SA→CA→FF
logits = linear_out(x)        # [B, L_m, vocab]
return logits
```

---
## 5.7 Phân tích tham số

| Khối                        | # Param |
|-----------------------------|---------|
| Token Embedding             | 64 k    |
| Text Projection (768→512)   | 393 k   |
| 6 × Transformer Encoder     | 24.7 M  |
| Output Linear (512→125)     | 64 k    |
| **Tổng**                    | ≈ 25.8 M|

Bộ nhớ FP16 ~1.1 GB (không tính gradient).

---
## 5.8 Ưu điểm – Hạn chế

**Ưu điểm**
* Cross-attention gắn kết mô tả và nhạc – phù hợp bài toán text-to-music.  
* Encoder-only ⇒ sinh nhanh, không cần decoder phức tạp.

**Hạn chế**
* Mặc định mono-track, chưa ràng buộc hoà thanh.
* Text >512 token sẽ bị cắt.  
* Chưa tích hợp scale/key embedding.

---
## 5.9 Hướng mở rộng

* Hierarchical PE theo bar/beat.  
* Thêm constraint hòa âm (Key embedding).  
* Fine-tune BERT trên mô tả âm nhạc.  
* RL với reward tonal-distance.

---
**Tóm tắt**: Mô hình sử dụng 6 lớp Transformer (512-dim) với cross-attention từ nhạc sang văn bản; huấn luyện bằng CE/Label-Smoothing/Focal để chuyển văn bản giàu ngữ nghĩa thành chuỗi token MIDI. 