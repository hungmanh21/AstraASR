from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torchaudio.functional as F

from .loader import AudioData


# ---------------------------------------------------------------------------
# Signal transforms  (each returns a new AudioData)
# ---------------------------------------------------------------------------

def resample(audio: AudioData, target_sr: int = 16000) -> AudioData:
    if audio.sample_rate == target_sr:
        return audio

    waveform = torch.from_numpy(audio.waveform)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # (1, N)

    resampled = F.resample(waveform, audio.sample_rate, target_sr)

    if audio.waveform.ndim == 1:
        resampled = resampled.squeeze(0)  # back to (N,)

    return AudioData(
        waveform=resampled.numpy(),
        sample_rate=target_sr,
        source=audio.source,
    )


def to_mono(audio: AudioData) -> AudioData:
    if audio.waveform.ndim == 1:
        return audio
    if audio.waveform.shape[0] == 1:
        waveform = audio.waveform.squeeze(0)
    else:
        waveform = audio.waveform.mean(axis=0)
    return AudioData(waveform=waveform, sample_rate=audio.sample_rate, source=audio.source)


def normalize(audio: AudioData, method: str = "peak") -> AudioData:
    """Normalize amplitude. method='peak' scales to [-1, 1]; 'rms' targets RMS=0.1."""
    waveform = audio.waveform.copy()

    if method == "peak":
        peak = np.abs(waveform).max()
        if peak > 1e-8:
            waveform /= peak
    elif method == "rms":
        rms = np.sqrt(np.mean(waveform ** 2))
        if rms > 1e-8:
            waveform *= 0.1 / rms
            np.clip(waveform, -1.0, 1.0, out=waveform)
    else:
        raise ValueError(f"Unknown normalization method: {method!r}")

    return AudioData(waveform=waveform, sample_rate=audio.sample_rate, source=audio.source)


def trim_silence(audio: AudioData, top_db: float = 30.0) -> AudioData:
    """Remove leading/trailing silence using an energy threshold."""
    import librosa

    assert audio.waveform.ndim == 1, "trim_silence requires mono audio (call to_mono first)"
    waveform, _ = librosa.effects.trim(audio.waveform, top_db=top_db)
    return AudioData(waveform=waveform, sample_rate=audio.sample_rate, source=audio.source)


# ---------------------------------------------------------------------------
# VAD-based chunking
# ---------------------------------------------------------------------------

@dataclass
class AudioChunk:
    waveform: np.ndarray  # float32, (samples,) mono
    sample_rate: int
    start: float           # start time in the original audio (seconds)
    end: float             # end time in the original audio (seconds)
    index: int
    source: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def n_samples(self) -> int:
        return self.waveform.shape[0]


def chunk_by_vad(
    audio: AudioData,
    max_duration: float = 30.0,
    min_speech_s: float = 0.25,
    min_silence_s: float = 0.3,
    overlap_s: float = 0.5,
) -> List[AudioChunk]:
    """Segment audio into chunks using Silero VAD.

    Speech segments are merged and split to respect max_duration.  Adjacent
    chunks get overlap_s seconds of context from their neighbour so ASR has
    no hard cuts at word boundaries.

    Requires 16 kHz mono audio.
    """
    assert audio.waveform.ndim == 1, "chunk_by_vad requires mono audio (call to_mono first)"
    assert audio.sample_rate == 16000, "Silero VAD requires 16 kHz audio (call resample first)"

    from silero_vad import get_speech_timestamps, load_silero_vad

    model = load_silero_vad()
    tensor = torch.from_numpy(audio.waveform)

    speech_ts = get_speech_timestamps(
        tensor,
        model,
        sampling_rate=16000,
        min_speech_duration_ms=int(min_speech_s * 1000),
        min_silence_duration_ms=int(min_silence_s * 1000),
        return_seconds=True,
    )

    if not speech_ts:
        return [
            AudioChunk(
                waveform=audio.waveform,
                sample_rate=audio.sample_rate,
                start=0.0,
                end=audio.duration,
                index=0,
                source=audio.source,
            )
        ]

    segments = _split_long_segments(speech_ts, max_duration)
    total = audio.duration
    sr = audio.sample_rate
    n = len(segments)

    chunks = []
    for i, (seg_start, seg_end) in enumerate(segments):
        start = max(0.0, seg_start - (overlap_s if i > 0 else 0.0))
        end = min(total, seg_end + (overlap_s if i < n - 1 else 0.0))
        chunks.append(
            AudioChunk(
                waveform=audio.waveform[int(start * sr) : int(end * sr)],
                sample_rate=sr,
                start=start,
                end=end,
                index=i,
                source=audio.source,
            )
        )

    return chunks


def _split_long_segments(speech_ts: list, max_duration: float) -> list:
    """Force-split any speech segment that exceeds max_duration."""
    segments = []
    for ts in speech_ts:
        s, e = ts["start"], ts["end"]
        while e - s > max_duration:
            segments.append((s, s + max_duration))
            s += max_duration
        segments.append((s, e))
    return segments
