from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

from ..audio.loader import load
from ..audio.transforms import normalize, resample, to_mono
from ..models.whisper import TranscriptionResult, WhisperModel

_WHISPER_SR = 16_000


@dataclass
class TranscribeConfig:
    model_name: str = "openai/whisper-large-v3"
    language: Optional[str] = "vi"   # None → auto-detect from audio
    task: str = "transcribe"          # "transcribe" | "translate"

    # Mode: "sequential" uses Whisper's 30s sliding window with hallucination
    # detection; "chunked" pre-splits audio and batches chunks for throughput.
    mode: str = "sequential"

    # Sequential parameters
    condition_on_prev_tokens: bool = True
    compression_ratio_threshold: float = 1.35
    logprob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    temperatures: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    # Chunked parameters
    chunk_length_s: float = 30.0
    stride_length_s: float = 5.0
    batch_size: int = 8

    device: Optional[str] = None


def transcribe(
    source: Union[str, Path, bytes, np.ndarray],
    config: Optional[TranscribeConfig] = None,
) -> TranscriptionResult:
    """Preprocess audio then transcribe with Whisper.

    Steps: load → resample to 16 kHz → mono → peak-normalize.
    The preprocessed waveform is handed directly to the HuggingFace
    pipeline; no VAD chunking is applied.
    """
    if config is None:
        config = TranscribeConfig()

    audio = load(source)
    audio = resample(audio, _WHISPER_SR)
    audio = to_mono(audio)
    audio = normalize(audio, method="peak")

    model = WhisperModel(model_name=config.model_name, device=config.device)

    if config.mode == "chunked":
        return model.transcribe_chunked(
            waveform=audio.waveform,
            sample_rate=_WHISPER_SR,
            language=config.language,
            task=config.task,
            chunk_length_s=config.chunk_length_s,
            stride_length_s=config.stride_length_s,
            batch_size=config.batch_size,
            source=str(source),
        )

    return model.transcribe_sequential(
        waveform=audio.waveform,
        sample_rate=_WHISPER_SR,
        language=config.language,
        task=config.task,
        condition_on_prev_tokens=config.condition_on_prev_tokens,
        compression_ratio_threshold=config.compression_ratio_threshold,
        logprob_threshold=config.logprob_threshold,
        no_speech_threshold=config.no_speech_threshold,
        temperatures=config.temperatures,
        source=str(source),
    )
