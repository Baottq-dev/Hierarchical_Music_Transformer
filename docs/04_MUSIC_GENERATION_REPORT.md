# 04 – Music Generation Report

> Inference pipeline, interactive tools, and qualitative analysis
>
> *AMT Project – v1.0 | Last updated: 2025-07-01*

---

## 1. Inference Modes

| Mode | Script/Class | Use-case |
|------|--------------|----------|
| **Batch** | `run_test.py` | Offline large-scale generation for evaluation |
| **Interactive** | `source.test.tester.InteractiveGenerator` | Notebooks, demos |
| **API** | `source.test.evaluator.FastAPI` | Web service deployment |

### 1.1 CLI Example

```bash
python run_test.py --config configs/generate.yaml --num_clips 64 --out_dir out/batch
```

### 1.2 Notebook Example

```python
from source.test.tester import InteractiveGenerator
ig = InteractiveGenerator('checkpoints/best.pt')
mid = ig(prompt='up-tempo latin jazz, piano solo', length=45)
mid.save('latin_jazz.mid')
ig.play(mid)  # IPython.display.Audio
```

---

## 2. Prompt Conditioning

* **Text embedding** – BERT CLS vector prepended to decoder input.
* **Genre tag** – one-hot token inserted after BOS.
* **Tempo hint** – scalar BPM encoded via learned embedding.

---

## 3. Temperature & Top-k / p Sampling

| Param | Typical | Effect |
|-------|---------|--------|
| Temperature | 0.9 | higher = more diverse |
| Top-k | 64 | limits unlikely tokens |
| Top-p | 0.95 | nucleus sampling |

Default combination: `temp=0.95, top_k=64, top_p=0.95`.

---

## 4. Speed Benchmark

| Device | Batch 8 × 30 s | tok/s | real-time factor |
|--------|---------------|-------|------------------|
| RTX 3060 12 GB | 3.1 s | 820 | 0.10× |
| RTX 3090 24 GB | **1.8 s** | 1 410 | 0.06× |
| CPU i7-12700K | 27.9 s | 83 | 0.86× |

*(Lower real-time factor better; 0.06× ⇒ 16× faster than real-time)*

---

## 5. Qualitative Evaluation

### 5.1 Tension Curve

![tension](assets/tension.png)

Generated pieces exhibit typical A-B-A song-form with climax near 70 % duration.

### 5.2 Human Listening Test

| Metric (1-5) | AMT | Music Transformer |
|--------------|-----|-------------------|
| Pleasantness | **4.1** | 3.6 |
| Coherence | **4.0** | 3.2 |
| Originality | 3.9 | **4.1** |

30 participants, Latin Square design.

---

## 6. Error Cases & Mitigation

| Issue | Symptom | Fix |
|-------|---------|-----|
| Mode collapse | repeated 2-bar riff | increase temp / enable repetition penalty |
| Empty output | only EOS | lower top-k / ensure prompt tokens not OOV |
| GPU OOM | long len | use KV-cache + generate in chunks |

---

## 7. Future Work

* Real-time accompaniment via incremental decoding.
* Diffusion-based post-processing for expressive timing.
* User-in-the-loop melody editing (MuseScore plugin).

---

*End of Report 04.* 