# 02 – Data Processing Report

> Symbolic-music tokenisation, textual feature engineering, and dataset preparation
>
> *AMT Project – v1.0 | Last updated: 2025-07-01*

---

## 1. Goals

1. Convert raw MIDI + text pairs into **model-ready tensors**.
2. Preserve musical expressiveness (velocities, timing nuances).
3. Generate rich side-features to enable future conditional generation.

---

## 2. MIDI Processing Pipeline

```mermaid
graph TD
  A[clean_midi/*.mid] --> B{MidiProcessor}
  B -->|pretty_midi| C[Note events]
  C --> D[Quantise 10 ms]
  D --> E[Split 30 s windows]
  E --> F[Event Encoder]
  F --> G[Token IDs]
  G --> H[torch.save("dataset.pt")]
```

### 2.1 Event Vocabulary (REMIGEN-v1)

| Token Type | Range | Example |
|------------|-------|---------|
| `PITCH_XX` | 0-127 | `PITCH_60` (C4) |
| `TIME_SHIFT_YY` | 10-1000 ms | `TIME_SHIFT_120` |
| `DUR_ZZ` | 10-2000 ms | `DUR_480` |
| `VEL_VV` | 1-127 | `VEL_90` |
| `BOS`, `EOS` | n/a | sequence boundaries |

Vocabulary size: **512** tokens.

### 2.2 Multi-Track Alignment

1. Merge drums to a single percussion track.
2. Instruments sorted by **MIDI program group** to enforce deterministic ordering.
3. Polyphony capped at 8 concurrent notes; extra notes pruned by lowest velocity.

### 2.3 Data Augmentations

* Pitch transposition ±5 semitones (except percussion).
* Velocity jitter ±8.
* Tempo scaling ×{0.9, 1.1}.

---

## 3. Text Processing Pipeline

```python
import spacy, transformers, sklearn
nlp = spacy.load("en_core_web_lg")
bert = transformers.AutoModel.from_pretrained("bert-base-uncased")

def process(text: str):
    doc = nlp(text)
    ent_map = {ent.label_: ent.text for ent in doc.ents}
    tfidf_vec = tfidf.transform([text])
    bert_vec = bert(text)[0][:,0,:]  # CLS token
    return {"tokens": [t.text for t in doc],
            "entities": ent_map,
            "tfidf": tfidf_vec,
            "bert": bert_vec}
```

### 3.1 Custom Entity Recognition

Fine-tuned 20-epoch spaCy model using 2 k manually-labelled sentences for labels: `GENRE`, `MOOD`, `INSTRUMENT` (F1 = 0.87).

### 3.2 Feature Store Structure

```
processed/
  midi/
    000001.pt  # dict(tokens, lengths, meta)
  text/
    000001.pkl  # dict(tokens, bert, tfidf, ent)
  index.csv     # maps pair_id → file names
```

---

## 4. Correlation Analysis

![corr_heat](assets/corr_heatmap.png)

* Positive corr: **tempo ↔️ sentiment_valence (ρ=0.42)**.
* Negative corr: **note_density ↔️ abstractness (ρ=-0.36)**.

---

## 5. Dataset Split

| Split | Pieces | Tokens | Comment |
|-------|--------|--------|---------|
| Train | 15 121 | 61 M | stratified by genre |
| Val   | 1 890  | 7.6 M | random |
| Test  | 1 891  | 7.5 M | genre-balanced |

---

## 6. Reproducibility & Caching

* Processing deterministic when `seed=1234`.
* Intermediate artefacts cached under `data/cache/` with automatic invalidation if code hash changes.

---

## 7. Future Enhancements

* **Chord extraction** and explicit `CHORD_XXX` tokens.
* Filter-and-refill sampling for long-tail genres.
* On-the-fly data streaming with WebDataset for large-scale training.

---

*End of Report 02.* 