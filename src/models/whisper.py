from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import pipeline as hf_pipeline


@dataclass
class TranscriptionSegment:
    text: str
    start: float    # seconds from start of original audio
    end: float
    language: str


@dataclass
class TranscriptionResult:
    text: str
    segments: List[TranscriptionSegment]
    language: str
    source: str
    duration: float


class WhisperModel:
    """Thin wrapper around HuggingFace automatic-speech-recognition pipeline.

    Two calling modes, both delegating fully to the pipeline:

    transcribe_sequential  — passes return_timestamps=True with no
    chunk_length_s.  The pipeline invokes Whisper's built-in 30-second
    sliding-window decoder (timestamp tokens, condition_on_prev_tokens,
    compression-ratio / log-prob hallucination detection, temperature
    fallback).

    transcribe_chunked  — additionally passes chunk_length_s and
    stride_length_s.  The pipeline pre-splits audio into overlapping
    fixed-length chunks before the encoder, then batches them together
    (batch_size) for higher GPU throughput.  Stride regions are decoded
    but masked out of the final merge; no hallucination detection.
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-large-v3",
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch_dtype is None:
            torch_dtype = torch.float16 if device != "cpu" else torch.float32

        self._device = device
        self._torch_dtype = torch_dtype
        self._pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=model_name,
            torch_dtype=torch_dtype,
            device=device,
        )

    def detect_language(self, waveform: np.ndarray, sample_rate: int = 16000) -> str:
        """Return ISO-639-1 language code detected from the first ≤30 s."""
        audio = waveform[: sample_rate * 30]
        features = self._pipe.feature_extractor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features.to(self._device, dtype=self._torch_dtype)

        with torch.no_grad():
            lang_ids = self._pipe.model.detect_language(features)

        lang_token = self._pipe.tokenizer.convert_ids_to_tokens([lang_ids[0].item()])[0]
        return lang_token[2:-2]  # "<|vi|>" → "vi"

    def transcribe_sequential(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        *,
        language: Optional[str] = None,
        task: str = "transcribe",
        condition_on_prev_tokens: bool = True,
        compression_ratio_threshold: float = 1.35,
        logprob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6,
        temperatures: Tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        source: str = "",
    ) -> TranscriptionResult:
        detected_lang = language or self.detect_language(waveform, sample_rate)
        output = self._pipe(
            {"array": waveform, "sampling_rate": sample_rate},
            return_timestamps=True,
            generate_kwargs={
                "language": detected_lang,
                "task": task,
                "condition_on_prev_tokens": condition_on_prev_tokens,
                "compression_ratio_threshold": compression_ratio_threshold,
                "logprob_threshold": logprob_threshold,
                "no_speech_threshold": no_speech_threshold,
                "temperature": temperatures,
            },
        )
        return self._parse_output(output, language=detected_lang, source=source)

    def transcribe_chunked(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        *,
        language: Optional[str] = None,
        task: str = "transcribe",
        chunk_length_s: float = 30.0,
        stride_length_s: float = 5.0,
        batch_size: int = 8,
        source: str = "",
    ) -> TranscriptionResult:
        detected_lang = language or self.detect_language(waveform, sample_rate)
        output = self._pipe(
            {"array": waveform, "sampling_rate": sample_rate},
            chunk_length_s=chunk_length_s,
            stride_length_s=stride_length_s,
            batch_size=batch_size,
            return_timestamps=True,
            generate_kwargs={"language": detected_lang, "task": task},
        )
        return self._parse_output(output, language=detected_lang, source=source)

    def _parse_output(
        self, output: dict, language: str, source: str
    ) -> TranscriptionResult:
        full_text = output.get("text", "").strip()
        segments: List[TranscriptionSegment] = []

        for chunk in output.get("chunks") or []:
            ts = chunk.get("timestamp") or (None, None)
            t_start = ts[0] if ts[0] is not None else 0.0
            t_end = ts[1] if ts[1] is not None else t_start
            text = chunk.get("text", "").strip()
            if text:
                segments.append(
                    TranscriptionSegment(
                        text=text, start=t_start, end=t_end, language=language
                    )
                )

        duration = segments[-1].end if segments else 0.0
        return TranscriptionResult(
            text=full_text,
            segments=segments,
            language=language,
            source=source,
            duration=duration,
        )
