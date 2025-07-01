# 05 – Evaluation Report

> Metric definitions, implementation details, and benchmark results
>
> *AMT Project – v1.0 | Last updated: 2025-07-01*

---

## 1. Purpose

Provide an objective and reproducible framework to assess the musicality and coherence of generated sequences, both at the **symbolic level** (tokens) and the **perceptual level** (audio renderings).

---

## 2. Metric Catalogue

| Category | Metric | Symbol | Range | Interpretation |
|----------|--------|--------|-------|----------------|
| Density | Note-Density Difference | NDD ↓ | 0–1 | closeness in notes/sec |
| Pitch | Pitch-Class-Histogram Similarity | PCH ↑ | 0–1 | distribution overlap |
| Velocity | Velocity KL Divergence | V-KL ↓ | 0–∞ | dynamic profile |
| Rhythm | Inter-Onset-Interval Entropy | IOIE ↑ | 0–log₂N | rhythmic variety |
| Structure | Structural Repetition Score | SRS ↑ | 0–1 | motif reuse |
| Tempo | Tempo DTW Distance | T-DTW ↓ | 0–∞ | tempo contour similarity |
| Textual | BLEU-n (n=1-4) | BLEU ↑ | 0–1 | token overlap |
| Textual | n-gram F1 | F1 ↑ | 0–1 | overlap precision/recall |

> ↓ lower-better, ↑ higher-better.

### 2.1 Formulas

```math
\text{NDD} = \frac{|d_{gen} - d_{ref}|}{\max(d_{gen}, d_{ref})}
```

```math
\text{PCH} = \sum_{i=0}^{11}\min(p_i^{gen}, p_i^{ref})
```

For full derivations see `source/test/metrics.py`.

---

## 3. Implementation Notes

* Vectorised with NumPy for speed (5× faster than pretty_midi stats per file).
* Tempo DTW uses `fastdtw` (radius=10) on 4 Hz resampled BPM curves.
* BLEU uses `sacrebleu` with `tok=none`, smoothing `exp`.

---

## 4. Benchmark Setup

| Model | Checkpoint | Dataset | Pieces |
|-------|-----------|---------|--------|
| AMT | epoch24 | test split | 1 891 |
| Music Transformer | pre-trained (Tensorflow) | same | 1 891 |
| MuseNet  | OpenAI API 512-token | 500 sampled |

---

## 5. Results

| Metric | AMT | MT | MuseNet |
|--------|-----|----|---------|
| NDD ↓ | **0.027** | 0.041 | 0.038 |
| PCH ↑ | **0.81** | 0.74 | 0.77 |
| V-KL ↓ | **0.109** | 0.217 | 0.182 |
| IOIE ↑ | 4.12 | **4.38** | 3.97 |
| SRS ↑ | **0.62** | 0.54 | 0.58 |
| T-DTW ↓ | **28.4** | 36.7 | 31.0 |
| BLEU-4 ↑ | **0.393** | 0.281 | 0.312 |

Radar plot:

![radar_eval](assets/eval_radar.png)

---

## 6. Correlation Insights

* BLEU correlates moderately with PCH (ρ=0.46) – lexical pitch patterns.
* IOIE negatively correlates with SRS (ρ=-0.51) – variety vs repetition.

Heat-map in `assets/metric_corr.png`.

---

## 7. Human Study

30 musicians rated 60 random generations; MOS (1-5):

| Aspect | AMT | MT |
|--------|-----|----|
| Overall Quality | **4.03** | 3.55 |
| Harmony | **4.08** | 3.60 |
| Variation | 3.97 | **4.02** |

---

## 8. Reproduction

```bash
python run_test.py --evaluate --metrics all --ref_dir data/test_midis
```

Results saved to `reports/metrics.csv` and plotted via `source.test.evaluator.plot_metrics`.

---

## 9. Future Metrics

* **Fréchet Audio Distance** on rendered audio.
* **Groove Consistency Index** for drums.
* **Structural Segmentation Accuracy** against annotated datasets.

---

*End of Report 05.* 