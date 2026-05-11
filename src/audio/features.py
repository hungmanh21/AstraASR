from dataclasses import dataclass
from typing import List, Optional

import torch
from transformers import WhisperProcessor

from .transforms import AudioChunk


@dataclass
class WhisperFeatures:
    input_features: torch.Tensor  # (1, 80, 3000)
    start: float
    end: float
    index: int
    source: str


class WhisperFeatureExtractor:
    """Wraps WhisperProcessor for batch feature extraction from AudioChunk objects.

    The processor is loaded once and reused across calls.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        device: Optional[str] = None,
    ):
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, chunks: List[AudioChunk]) -> List[WhisperFeatures]:
        return [self._extract(chunk) for chunk in chunks]

    def _extract(self, chunk: AudioChunk) -> WhisperFeatures:
        inputs = self.processor(
            chunk.waveform,
            sampling_rate=chunk.sample_rate,
            return_tensors="pt",
        )
        return WhisperFeatures(
            input_features=inputs.input_features.to(self.device),
            start=chunk.start,
            end=chunk.end,
            index=chunk.index,
            source=chunk.source,
        )
