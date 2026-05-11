from dataclasses import dataclass, fields
from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from .loader import AudioData, load
from .transforms import (
    AudioChunk,
    chunk_by_vad,
    normalize,
    resample,
    to_mono,
    trim_silence,
)


@dataclass
class AudioPipelineConfig:
    target_sr: int = 16000
    normalize_method: str = "peak"    # "peak" | "rms"
    do_trim_silence: bool = True
    trim_top_db: float = 30.0
    vad_max_duration: float = 30.0    # max seconds per chunk (Whisper limit)
    vad_min_speech_s: float = 0.25    # min speech segment length for VAD
    vad_min_silence_s: float = 0.3    # min silence gap to trigger a split
    chunk_overlap_s: float = 0.5      # overlap context added at chunk boundaries

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AudioPipelineConfig":
        """Load config from a YAML file, ignoring unknown keys."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def run(
    source: Union[str, Path, bytes, np.ndarray],
    config: Optional[AudioPipelineConfig] = None,
    sample_rate: int = 16000,
) -> List[AudioChunk]:
    """Full audio preprocessing pipeline.

    Steps
    -----
    1. Load  — decode any source (file, URL, YouTube, bytes, array) to float32 PCM
    2. Resample  — standardise to target_sr (default 16 kHz)
    3. Mono  — average channels if stereo/multi-channel
    4. Normalize  — scale amplitude (peak or RMS)
    5. Trim silence  — strip leading/trailing silence (optional)
    6. VAD chunk  — segment into ≤30 s speech chunks with overlap

    Returns a list of AudioChunk objects ready for ASR model input.
    """
    if config is None:
        config = AudioPipelineConfig()

    audio: AudioData = load(source, sample_rate=sample_rate)
    audio = resample(audio, config.target_sr)
    audio = to_mono(audio)
    audio = normalize(audio, method=config.normalize_method)

    if config.do_trim_silence:
        audio = trim_silence(audio, top_db=config.trim_top_db)

    return chunk_by_vad(
        audio,
        max_duration=config.vad_max_duration,
        min_speech_s=config.vad_min_speech_s,
        min_silence_s=config.vad_min_silence_s,
        overlap_s=config.chunk_overlap_s,
    )
