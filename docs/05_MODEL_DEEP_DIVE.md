# 5. PHÂN TÍCH SÂU KIẾN TRÚC MÔ HÌNH

> **Mục tiêu**: Giải thích chi tiết cách `MusicTransformer` chuyển đổi cặp *(Text, MIDI-history)* thành **token nhạc kế tiếp**, kèm lập luận thiết kế, công thức và phân tích độ phức tạp.

---
## 5.1 Cấu trúc tầng cao

```
                ┌─────────────────────────────┐
                │  Text Feature Projection    │  (Linear 768→512 + PE)
Input Text ───▶ │  T ∈ ℝ^{L_t×512}            │
                └──────────────┬──────────────┘
                               │ (Cross-Attn K,V)
               ┌───────────────▼─────────────────┐
MIDI Tokens ─▶ │  Music Encoder  (6× blocks)     │  M^0 ∈ ℝ^{L_m×512}
 (shifted)     │  SA + CA + FF + LN             │
               └───────────────┬─────────────────┘
                               │   (Self-Attn)
               ┌───────────────▼─────────────────┐
               │  Output Projection W_o          │  logits ∈ ℝ^{L_m×V}
               └─────────────────────────────────┘
```

* **SA**: Self-Attention trong chuỗi nhạc.  
* **CA**: Cross-Attention: Query từ nhạc, Key/Value từ text.

---
## 5.2 Thành phần chi tiết

### 5.2.1 Embedding & Positional Encoding
* **Token Embedding** \(E\_m ∈ ℝ^{V×d}\), `V = 125`, `d = 512`.  
* **Positional** (sinusoidal) \(P^{(m)} \in ℝ^{L_m×d}\), \(P^{(t)} \in ℝ^{L_t×d}\).

\[
M^0 = E\_m[y] + P^{(m)}, \quad T = (W_p x) + P^{(t)}
\]
*`W_p`* là ma trận 768×512.

### 5.2.2 Khối Transformer (layer j)

```
M^j = LN( M^{j-1} + SA(M^{j-1}) )
if use_cross:
    M^j = LN( M^j + CA(M^j, T) )
M^j = LN( M^j + FF(M^j) )
```
* **SA**: Multi-Head (h=8) với `d_k = d/h = 64`.  
* **CA**: giống SA nhưng K,V = Linear(T).  
* **FF**: `d_ff = 2048`, ReLU.

Time/space per layer:  
`O(L_m² d)` for SA, `O(L_m L_t d)` for CA (≈ 1024×512×512 ≈ 0.27M mult per head).

### 5.2.3 Cross-Attention chi tiết

\[
Q = M W_q \in ℝ^{L_m×d},\; K,V = T W_k, T W_v \in ℝ^{L_t×d}
\]

Score: \(S = \frac{QK^⊤}{\sqrt d}\)  
Mask (optional) để bỏ PAD trong text.

Context: \(C = \text{softmax}(S) V\)  
Output: \(CA(M,T) = W_o C\).

**Tại sao chọn Query=midi, Key/Value=text?**  
Bởi vì mục tiêu là điều hoà **chuỗi nhạc** dựa trên đặc trưng text; MIDI cần "hỏi" text để lấy bối cảnh.

---
## 5.3 Mất mát & Regularisation

* **Padding Mask**: token=0 bỏ qua loss & attention.  
* **Label-smoothing** ε=0.1 (nếu bật) → tránh over-confidence.  
* **Dropout** 0.1 mọi linear & attention.  
* **Gradient Clip** 1.0.

---
## 5.4 Luồng dữ liệu (Forward) – mã giả
```python
def forward(midi_seq, text_emb):
    midi_emb = token_emb(midi_seq) + pos_midi
    text_proj = linear(text_emb) + pos_text

    x = midi_emb
    for l in layers:
        x = ln(x + self_attn(x))
        if cross:
            x = ln(x + cross_attn(x, text_proj))
        x = ln(x + feed_forward(x))
    logits = output_linear(x)
    return logits
```

---
## 5.5 Phân tích tham số
| Khối | Công thức | #Param |
|------|-----------|--------|
| Token Emb | 125×512 | 64k |
| Text Projection | 768×512 | 393k |
| Encoder (×6) | 6×(4d² + 2d d_ff + d_ff d) | ≈ 24.7M |
| Output Linear | 512×125 | 64k |
| **Tổng** | 25.8M |

Chiếm GPU ≈ 1.1 GB (fp16) + gradient.

---
## 5.6 Ưu điểm và hạn chế
### Ưu điểm
1. Cross-attention cho phép liên kết mạnh giữa mô tả và nhạc, đặc biệt về **emotion** & **instrument keyword**.
2. Kiến trúc encoder-only → song song cao, inference nhanh (không cần decoder). 512-token batch sinh 2.3 it/s trên RTX 3090.

### Hạn chế
1. Độ dài text cố định 512 – câu cực dài bị cắt.  
2. Chỉ sinh **mono-track** (đa âm nhưng 1 nhạc cụ).  
3. Không ràng buộc nhạc lý (scale, chord progression) → đôi khi nốt "lạc tông".

---
## 5.7 Ý tưởng mở rộng
* **Hierarchical Positional Encoding** (bar, beat) để giữ cấu trúc dài.  
* **Key/Scale embedding**: học ràng buộc âm giai.  
* **Diffusion-based decoder** để cải thiện chi tiết dynamics.  
* **Reinforcement Learning** với reward Tonal-Distance.

---
**TL;DR**: Mô hình là một **Transformer encoder sâu 6 lớp**, thêm **cross-attention** để "hỏi" embedding văn bản, huấn luyện bằng CE/Label-Smooth/Focal, phát huy khả năng "dịch" mô tả ngữ nghĩa thành chuỗi token MIDI. 