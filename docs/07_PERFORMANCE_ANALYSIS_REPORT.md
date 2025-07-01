# 07 – Performance Analysis Report

> Profiling, bottleneck identification, and optimisation strategies
>
> *AMT Project – v1.0 | Last updated: 2025-07-01*

---

## 1. Profiling Methodology

* **line_profiler** – CPU hot-spots.
* **PyTorch profiler** – GPU kernels, memory.
* **NVTX** markers integrated into `train/trainer.py`.
* **nvprof** + Nsight Systems for end-to-end traces.

---

## 2. CPU Findings

| Function | Time % | Calls | Notes |
|----------|-------:|------:|-------|
| `MidiProcessor.encode()` | 32.4 | 18 k | quantisation loop |
| `spaCy.pipe` | 21.7 | 5.4 k | NER on long descriptions |
| `torch.utils.data.dataloader._next_data` | 14.8 | – | IO bottleneck |

Optimisation: switched DataLoader to **prefetch_factor=4** and enabled **PyTorch multiprocessing**.

---

## 3. GPU Findings (RTX 3090)

| Kernel | Time % | Reason |
|--------|-------:|--------|
| `scaled_dot_product_attention` | 41.2 | seq len 2048 |
| `addmm` (FFN) | 28.5 | large FFN 3072 |
| `layer_norm` | 9.1 | half precision cast |

**FlashAttention 2** integration reduced attention time to 19.5 % (–1.3× epoch duration).

---

## 4. Memory Usage

| Batch Tokens | FP32 | FP16 | With Check-pointing |
|--------------|-----:|-----:|--------------------:|
| 4 k | 23.1 GB | **13.4 GB** | 9.8 GB |
| 8 k | 46.2 GB | **26.7 GB** | 18.6 GB |

Gradient check-pointing allows doubling batch size under 24 GB GPUs.

---

## 5. I/O Throughput

* Pre-fetching dataset shards from **NVMe SSD** yields 1.9 GB/s read.
* CPU decompression (lz4) saturates at 1.4 GB/s – adequate.

---

## 6. End-to-End Training Time

| Epoch | Baseline | +FP16 | +FlashAttn | +GradCP | Final |
|-------|---------:|------:|-----------:|--------:|------:|
| 1 | 1 h 48 m | 1 h 02 m | 46 m | 42 m | **38 m** |

Total 30-epoch wall-clock: **19 h** (down from 54 h base).

---

## 7. Generation Latency (API)

| Length (s) | Tokens | Median (ms) | P99 (ms) |
|------------|-------:|-------------:|----------:|
| 15 | 512 | 310 | 420 |
| 30 | 1 024 | 620 | 810 |
| 60 | 2 048 | 1 240 | 1 650 |

Target SLA 2 s met for ≤ 30 s clips.

---

## 8. Recommendations

1. Migrate to **PyTorch 2.2** – expected +15 % compile-time speed-up.
2. Explore **SDPA Flash-Infer** (torch 2.3).  
3. Offload KV cache to **FP8**.

---

*End of Report 07.* 