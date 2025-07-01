# 06 – System Architecture Report

> End-to-end software and infrastructure design
>
> *AMT Project – v1.0 | Last updated: 2025-07-01*

---

## 1. Component Breakdown

| Layer | Technology | Description |
|-------|------------|-------------|
| Data Collection | Python asyncio, `aiohttp` | Parallel HTTP downloads |
| Processing | PyTorch, spaCy | CPU/GPU feature extraction |
| Model Training | PyTorch Lightning | Distributed FP16 training |
| Inference API | FastAPI, Uvicorn | Exposes `/generate` endpoint |
| Storage | AWS S3 / local FS | MIDI artefacts & checkpoints |
| Orchestration | Docker Compose | Local dev multi-container |
| CI/CD | GitHub Actions | Lint + test + push image |

---

## 2. Container Topology

```mermaid
graph LR
  subgraph docker-compose
    collect((collect))
    process((process))
    train((train))
    api((fastapi))
    db[(PostgreSQL)]
    storage[(MinIO S3)]
  end
  collect --> storage
  process --> storage
  train --> storage
  api --> db
  api --> storage
```

File `docker/docker-compose.yml` defines five services with distinct images, unified via shared `.env`.

---

## 3. Development vs Production

| Aspect | Dev (Compose) | Prod (K8s) |
|--------|---------------|------------|
| Orchestrator | docker-compose | EKS Kubernetes |
| Scalability | manual | HPA auto-scale |
| Storage | Local volumes | EFS + S3 |
| Logging | stdout | CloudWatch + Loki |

---

## 4. Sequence Diagram (Generation)

```mermaid
sequenceDiagram
  participant User
  participant FE as Frontend React
  participant BE as FastAPI
  participant Worker as Generator Pod
  participant S3

  User->>FE: click "Generate"
  FE->>BE: POST /generate {prompt, params}
  BE->>Worker: publish job (Redis)
  Worker->>S3: fetch checkpoint
  Worker-->>BE: job_id, status=running
  BE-->>FE: 202 Accepted
  Worker-->>S3: upload midi
  Worker-->>BE: status=done, url
  BE-->>FE: SSE progress
  FE-->>User: show download link
```

---

## 5. Security & Compliance

* **Rate limiting** – 10 req/min per IP (FastAPI middleware).
* **Auth** – JWT Bearer for private endpoints.
* **Licensing** – All generated MIDI released under CC-BY.
* **Data Privacy** – No personal data stored; S3 buckets encrypted with SSE-S3.

---

## 6. Monitoring & Observability

* **Prometheus** – metrics on GPU utilisation, latency.
* **Grafana** – dashboards (`docker/grafana/provisioning/`).
* **Loki** – central log aggregation.

Alert: P99 latency > 4 s triggers Slack webhook.

---

## 7. Future Enhancements

* Canary deployments for new checkpoints.
* GPU auto-scaling via Karpenter.
* Multi-tenant quotas.

---

*End of Report 06.* 