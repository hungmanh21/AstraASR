# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Astra-ASR** — Automatic Speech Recognition for Vietnamese, English, and code-switched (Viet–English) speech. Targets offline batch transcription (Phase 1) and online real-time transcription (Phase 2).

Primary model: Whisper large-v3 (HuggingFace Transformers). Secondary: NVIDIA NeMo Conformer. Requires CUDA GPU (~10 GB VRAM for Whisper large-v3).

## Setup

```bash
pip install uv
uv sync
```

PyTorch is pulled from the CUDA 12.6 index (`https://download.pytorch.org/whl/cu126`), configured in `pyproject.toml` under `[tool.uv.sources]`.

## Source Layout

```
src/
├── audio/              # Pre-processing, format conversion, VAD-based chunking  ✅ implemented
│   ├── __init__.py     # Re-exports all public symbols
│   ├── loader.py       # AudioData dataclass + load() — file/URL/YouTube/bytes/array
│   ├── transforms.py   # resample, to_mono, normalize, trim_silence, chunk_by_vad, AudioChunk
│   ├── features.py     # WhisperFeatureExtractor + WhisperFeatures (wraps WhisperProcessor)
│   └── pipeline.py     # AudioPipelineConfig + run() — full 6-step preprocessing pipeline
├── models/             # Whisper and NeMo wrappers  (not yet implemented)
├── inference/          # Batch (offline) and streaming (online) pipelines  (not yet implemented)
├── training/           # Fine-tuning scripts, W&B logging (train_whisper.py)  (not yet implemented)
├── evaluation/         # WER/CER/RTF metrics using jiwer + evaluate  (not yet implemented)
├── serving/            # FastAPI app (app.py), endpoints, Dockerfile  (not yet implemented)
└── data_pipeline/      # Collection, cleaning, annotation scripts  (not yet implemented)
```

### `src/audio` — implemented API

| Symbol | Module | Description |
|---|---|---|
| `AudioData` | `loader` | Dataclass: `waveform` (float32 ndarray), `sample_rate`, `source`; props: `duration`, `n_channels`, `n_samples` |
| `load(source)` | `loader` | Decode file / HTTP URL / YouTube URL / bytes / ndarray → `AudioData` |
| `resample(audio, target_sr)` | `transforms` | Resample via `torchaudio.functional.resample` |
| `to_mono(audio)` | `transforms` | Average channels → mono (1-D) waveform |
| `normalize(audio, method)` | `transforms` | `"peak"` (scale to ±1) or `"rms"` (target RMS=0.1) |
| `trim_silence(audio, top_db)` | `transforms` | Strip leading/trailing silence via `librosa.effects.trim` |
| `chunk_by_vad(audio, ...)` | `transforms` | Silero VAD segmentation → `List[AudioChunk]` with overlap |
| `AudioChunk` | `transforms` | Dataclass: `waveform`, `sample_rate`, `start`, `end`, `index`, `source` |
| `AudioPipelineConfig` | `pipeline` | Dataclass with all pipeline knobs; supports `from_yaml()` |
| `run(source, config)` | `pipeline` | End-to-end: load → resample → mono → normalize → trim → VAD chunk |
| `WhisperFeatureExtractor` | `features` | Wraps `WhisperProcessor`; `__call__(chunks)` → `List[WhisperFeatures]` |
| `WhisperFeatures` | `features` | Dataclass: `input_features` (torch Tensor), `start`, `end`, `index`, `source` |

CLI entry points (not yet implemented):
- `python -m astra_asr.infer --input <file|url> --lang <vi|en|auto>`
- `uvicorn src.serving.app:app --host 0.0.0.0 --port 8000`

## Key Design Decisions

- **Long audio**: VAD-based segmentation (Silero VAD) + overlap merging for arbitrarily long files
- **YouTube input**: `yt-dlp` handles audio extraction from URLs
- **Source separation**: `demucs` for separating foreground speech from background
- **Evaluation**: WER and CER via `jiwer`; RTF measured as processing_time / audio_duration (target RTF ≤ 0.3× on A100 80 GB)
- **Experiment tracking**: Weights & Biases (W&B)
- **Dataset format**: HuggingFace `datasets` format, pushed to HF Hub

## Evaluation Metrics

- **WER**: Word Error Rate = (S + D + I) / N — use CER for Vietnamese (character-level more meaningful)
- **RTF**: Real-Time Factor — must be < 1.0; target ≤ 0.3× on A100 80 GB for 60-min files
- **Error buckets to track**: missed English words, wrong Vietnamese diacritics, slang normalisation, hallucination, repeated text, timestamp drift, background speech confusion, speaker overlap

## Phase 1 Roadmap (Weeks 1–7)

1. Baseline pipeline + long-audio handling  ✅ `src/audio/` complete; tests in `tests/test_audio_pipeline.py`
2. Data construction pipeline (target: ≥ 100h clean audio on HF Hub)
3. Baseline evaluation + error analysis
4. Fine-tuning v1 (target: ≥ 10% relative WER improvement)
5. Fine-tuning v2 + targeted data improvement
6. FastAPI serving + Docker
7. Model optimization (INT8/FP16 via CTranslate2/faster-whisper; target: ≥ 2× RTF improvement, ≤ 1% WER degradation, fits 24 GB GPU)
