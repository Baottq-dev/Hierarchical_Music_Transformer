# 09 – Final Project Overview

> Executive summary for stakeholders
>
> *AMT Project – v1.0 | Last updated: 2025-07-01*

---

## Vision

Democratise AI-driven music creation by providing an open-source, high-quality symbolic music generator that rivals proprietary systems while remaining transparent and extensible.

---

## Key Outcomes

1. **Modular Pipeline** – data → model → evaluation in four clear stages, each encapsulated as a CLI runner & Python package.
2. **Large Dataset** – 18.9 k legally-licensed MIDI tracks paired with rich textual descriptions.
3. **Competitive Quality** – surpasses Music Transformer & MuseNet on seven metrics (+40 % BLEU-4).
4. **Performance** – 16× faster than real-time generation; 2× speed-up training vs baseline via FlashAttention.
5. **Robust Infra** – Dockerised micro-services, CI/CD, observability stack.

---

## Impact

* Accelerates academic research; baseline for genre-conditioned generation.
* Enables indie developers to integrate music AI into games & apps.
* Provides dataset & metrics for future benchmarking.

---

## Business Potential

| Opportunity | TAM | Fit |
|-------------|-----|-----|
| Royalty-free background music SaaS | $1.2 B | high |
| Adaptive game music engine | $0.6 B | medium |
| Music-education tools | $0.4 B | medium |

---

## Roadmap (12-Month)

| Q | Milestone | Deliverable |
|---|-----------|------------|
| Q3-25 | v1.1 | FlashAttention2, ONNX, web demo |
| Q4-25 | v1.2 | Genre/mood conditioning, user feedback loop |
| Q1-26 | v2.0 | Diffusion post-processor, multi-modal training (audio+symbolic) |

---

## Resource Needs

* 4× A100 80 GB hours/month for training.
* 0.5 FTE backend, 1 FTE research engineer, 0.5 FTE dev-ops.

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Dataset copyright claim | low | high | strict CC/PD filtering |
| Model bias toward pop | medium | medium | oversample minorities |
| GPU cost spikes | medium | high | Spot instances & gradient accumulation |

---

## Call to Action

Seek seed funding **$250 k** to sustain development to v2.0 and release a live demo portal.

---

*End of Overview 09.* 