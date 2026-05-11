# Astra-ASR

![Python](https://img.shields.io/badge/python-3.12+-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-in%20development-orange)

Automatic Speech Recognition system for Vietnamese, English, and code-switched (Viet–English) speech. Supports both **offline batch transcription** from files and URLs, and **online real-time transcription** for conversations and live talks.

---

## Overview

| Mode | Description |
|------|-------------|
| **Offline** | Transcribe audio/video files (MP3, MP4) or YouTube links; handles arbitrarily long audio via chunked processing |
| **Online** | Real-time transcription for talk shows and daily conversations mixing Vietnamese and English |

Target languages: **Vietnamese** (north / central / south accents), **English**, and **code-switched** Viet–English.

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Primary model | [Whisper large-v3](https://github.com/openai/whisper) (transformer-based) |
| Secondary model | [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) Conformer (optional, CTC/attention hybrid) |
| Language / runtime | Python 3.12 |
| Serving | FastAPI + Uvicorn |
| Containerisation | Docker |
| Experiment tracking | Weights & Biases (W&B) |
| Dataset hosting | Hugging Face Hub |

---

## Repository Structure

```
astra_asr/
├── data/                   # raw, processed, annotated datasets
│   ├── raw/
│   ├── processed/
│   └── annotations/
├── src/
│   ├── audio/              # audio pre-processing & chunking utilities
│   ├── models/             # model wrappers (Whisper, NeMo)
│   ├── inference/          # batch & streaming inference pipelines
│   ├── training/           # fine-tuning scripts & configs
│   ├── evaluation/         # metrics (WER, CER, RTF) & error analysis
│   ├── serving/            # FastAPI app, endpoints, Dockerfile
│   └── data_pipeline/      # collection, cleaning, annotation scripts
├── scripts/                # one-off helper scripts
├── notebooks/              # exploratory analysis
├── tests/
├── main.py
├── pyproject.toml
└── README.md
```

---

## Setup

**Requirements:** Python 3.12+, CUDA-capable GPU recommended (Whisper large-v3 requires ~10 GB VRAM).

```bash
# Clone
git clone <repo-url> && cd astra_asr

# Install with uv (recommended)
pip install uv
uv sync

# Or with pip
pip install -e .
```

### Key dependencies (to be added to pyproject.toml)

**Phase 1 – Offline**
```
openai-whisper / faster-whisper
nemo_toolkit[asr]
yt-dlp          # YouTube audio extraction
ffmpeg-python
torchaudio
transformers
datasets
evaluate        # WER / CER metrics
wandb
fastapi
uvicorn
```

---

## Usage

> These examples reflect the intended CLI/API — implementation in progress.

### Offline — transcribe a file

```bash
python -m astra_asr.infer --input audio.mp3 --lang vi
python -m astra_asr.infer --input video.mp4 --lang auto
```

### Offline — transcribe a YouTube URL

```bash
python -m astra_asr.infer --input "https://youtube.com/watch?v=..." --lang vi
```

### Online — real-time streaming (Phase 2)

```bash
python -m astra_asr.stream --device mic --lang auto
```

### REST API (Phase 1 Week 6+)

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/transcribe/short` | POST | Audio ≤ 30 s |
| `/transcribe/long` | POST | Audio > 30 s (chunked) |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus-compatible metrics |

---

## Roadmap

### Phase 1 — Offline Transcription (Weeks 1–7)

---

#### Week 1 — Repo Setup + Baseline Pipeline + Long-Audio Handling

**Goal:** End-to-end pipeline that accepts an audio/video file or URL and outputs a transcript. Must handle long audio at reasonable inference speed.

**Tasks**
- [ ] Finalize repo structure (folders, linting, CI skeleton)
- [ ] Audio pre-processing pipeline — format conversion, noise gate, sample-rate normalisation
- [ ] Long-audio chunking strategy (VAD-based segmentation + overlap merging)
- [ ] Baseline batch inference — Whisper large-v3
- [ ] Baseline batch inference — NeMo Conformer (optional)
- [ ] End-to-end smoke test on a 60-minute Vietnamese podcast

**Success criteria**
- Pipeline runs without errors on MP3, MP4, and a YouTube URL
- Whisper inference RTF ≤ 0.3× on a single A100 80 GB for a 60-min file

---

#### Week 2 — Data Construction Pipeline

**Goal:** Reproducible pipeline from raw audio sources → cleaned, annotated, HF-ready dataset.

**Tasks**
- [ ] Survey open datasets on HF Hub suitable for Vietnamese ASR (e.g., VLSP, CommonVoice vi, PhoWhisper eval sets)
- [ ] Identify high-quality crawl sources (YouTube channels, podcasts) covering north / central / south accents
- [ ] Data collection pipeline: URL list → download raw audio + metadata
- [ ] Data processing & cleaning: silence trimming, loudness normalisation, deduplication
- [ ] Annotation pipeline: auto-label with baseline Whisper → human review workflow
- [ ] Convert to HF `datasets` format and push to Hub

**Success criteria**
- ≥ 100 h clean Vietnamese + code-switched audio collected
- Dataset card written and pushed to HF Hub

---

#### Week 3 — Baseline Evaluation + Error Analysis

**Goal:** Quantify baseline model quality; identify top failure modes to guide data and fine-tuning strategy.

**Tasks**
- [ ] Implement WER / CER evaluation using `evaluate` library
- [ ] Implement RTF (Real-Time Factor) measurement
- [ ] Run evaluation on collected dataset (held-out test split)
- [ ] Bucket errors and compute per-bucket rates (see [Error Analysis Buckets](#error-analysis-buckets))
- [ ] Write analysis report with examples per error bucket

**Success criteria**
- Baseline WER/CER numbers documented for Vietnamese and English subsets
- Top 3 error buckets identified with supporting examples

---

#### Week 4 — Fine-tuning v1

**Goal:** Fine-tune Whisper large-v3 on the constructed dataset; evaluate improvement.

**Tasks**
- [ ] Training script (`src/training/train_whisper.py`) with gradient checkpointing
- [ ] Evaluation callback during training (WER on dev set)
- [ ] W&B logging integration (loss, WER, LR, grad norm)
- [ ] Run fine-tuning and evaluate on held-out test set
- [ ] Error analysis pass (same buckets as Week 3)

**Success criteria**
- Relative WER improvement ≥ 10% over baseline on Vietnamese test set
- Training run logged in W&B with reproducible config

---

#### Week 5 — Fine-tuning v2 + Targeted Data Improvement

**Goal:** Address the specific weaknesses found in Week 4 error analysis by adding targeted data and re-fine-tuning.

**Tasks**
- [ ] Identify top 2–3 error buckets still underperforming after v1
- [ ] Collect or synthesise additional data for those buckets (e.g., more code-switched examples, diverse accent data)
- [ ] Re-run fine-tuning with augmented dataset
- [ ] Evaluate v2 vs v1 on full test set and per-bucket

**Success criteria**
- Targeted error bucket rates improve ≥ 15% relative vs v1
- Overall WER does not regress

---

#### Week 6 — Production Serving

**Goal:** Deployable REST API service with health checks and basic monitoring.

**Tasks**
- [ ] FastAPI app skeleton (`src/serving/app.py`)
- [ ] `/transcribe/short` endpoint (audio ≤ 30 s, synchronous)
- [ ] `/transcribe/long` endpoint (audio > 30 s, chunked async)
- [ ] `/health` endpoint
- [ ] `/metrics` endpoint (request count, latency p50/p99)
- [ ] Dockerfile (multi-stage build, CUDA base image)
- [ ] Basic load test (verify latency SLA at 1 concurrent request)

**Success criteria**
- Docker image builds and runs; all endpoints return correct responses
- Short-audio p99 latency ≤ 2 s on test hardware

---

#### Week 7 — Model Optimization & Inference Enhancement

**Goal:** Reduce latency and memory footprint for production use.

**Tasks**
- [ ] Benchmark INT8 / FP16 quantisation (CTranslate2 / faster-whisper) vs FP32 baseline
- [ ] Evaluate dynamic batching for the long-audio endpoint
- [ ] Profile GPU utilisation and identify bottlenecks
- [ ] Apply best optimisations and re-validate WER / RTF

**Success criteria**
- RTF improvement ≥ 2× vs unoptimised baseline with ≤ 1% relative WER degradation
- Memory footprint reduced to fit within a single 24 GB GPU

---

### Phase 2 — Online Transcription (Planned, TBU)

Real-time transcription pipeline for live Vietnamese–English conversations.

| Area | Plan |
|------|------|
| Streaming ASR | Integrate faster-whisper or NeMo streaming decoder |
| Voice activity detection | Silero VAD or WebRTC VAD for endpoint detection |
| Speaker diarisation | Optional: pyannote.audio |
| Latency target | < 500 ms word-level latency |
| Transport | WebSocket API |
| Frontend | Minimal React / plain JS demo UI |

---

## Evaluation Metrics

| Metric | Definition |
|--------|-----------|
| **WER** | Word Error Rate = (S + D + I) / N — substitutions, deletions, insertions over total reference words |
| **CER** | Character Error Rate — same formula at character level; better suited for Vietnamese |
| **RTF** | Real-Time Factor = processing time / audio duration; RTF < 1.0 means faster than real-time |

---

## Error Analysis Buckets

| Bucket | Description |
|--------|-------------|
| Missed English word | Code-switched English token transcribed incorrectly or dropped |
| Wrong Vietnamese diacritics | Tonal mark or vowel nucleus error (e.g., "hoa" vs "hoà") |
| Slang normalisation | Informal word not mapped to standard form |
| Hallucination | Model generates plausible-sounding but entirely fabricated text |
| Repeated text | Segment duplicated due to chunking overlap or decoder loop |
| Timestamp drift | Segment boundaries misaligned with actual speech by > 500 ms |
| Background speech confusion | Background talker transcribed instead of (or merged with) foreground |
| Speaker overlap | Two simultaneous speakers produce garbled or merged transcript |

---

## Contributing

This is currently an internal research project. Contributions and feedback are welcome — open an issue or contact the maintainer.

---

*Astra-ASR — built for real-world Vietnamese and bilingual speech.*
