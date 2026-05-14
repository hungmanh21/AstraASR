"""
Tests for src/inference transcription pipeline.

Unit tests use a synthetic waveform and mock the HF pipeline so no model
weights are required.  Integration tests (marked with @pytest.mark.integration)
load real model weights and require a CUDA GPU.

Run unit tests only:   uv run pytest tests/test_inference.py -v
Run integration tests: uv run pytest tests/test_inference.py -v -m integration
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.inference.transcribe import TranscribeConfig, transcribe
from src.models.whisper import TranscriptionResult, TranscriptionSegment, WhisperModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 16_000


def _sine(duration_s: float = 5.0, sr: int = SR) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 0.5).astype(np.float32)


def _fake_pipeline_output(texts=("hello world",)):
    """Minimal dict matching HF pipeline return format."""
    chunks = []
    t = 0.0
    for text in texts:
        end = t + 2.0
        chunks.append({"text": f" {text}", "timestamp": (t, end)})
        t = end
    return {"text": " ".join(texts), "chunks": chunks}


# ---------------------------------------------------------------------------
# Unit tests (no model weights)
# ---------------------------------------------------------------------------

class TestWhisperModel:
    def _model_with_mock_pipe(self, pipe_output: dict) -> WhisperModel:
        with patch("src.models.whisper.hf_pipeline") as mock_hf:
            mock_pipe = MagicMock()
            mock_pipe.return_value = pipe_output
            mock_pipe.feature_extractor = MagicMock()
            mock_pipe.tokenizer = MagicMock()
            mock_pipe.model = MagicMock()
            mock_hf.return_value = mock_pipe
            model = WhisperModel(model_name="openai/whisper-tiny", device="cpu")
        return model

    def test_parse_output_maps_segments(self):
        output = _fake_pipeline_output(("xin chào", "thế giới"))
        model = self._model_with_mock_pipe(output)
        result = model._parse_output(output, language="vi", source="test")

        assert result.language == "vi"
        assert result.source == "test"
        assert len(result.segments) == 2
        assert result.segments[0].text == "xin chào"
        assert result.segments[0].start == pytest.approx(0.0)
        assert result.segments[0].end == pytest.approx(2.0)
        assert result.segments[1].start == pytest.approx(2.0)
        assert result.duration == pytest.approx(4.0)

    def test_parse_output_skips_empty_text(self):
        output = {"text": "", "chunks": [{"text": "  ", "timestamp": (0.0, 1.0)}]}
        model = self._model_with_mock_pipe(output)
        result = model._parse_output(output, language="en", source="")
        assert result.segments == []
        assert result.duration == pytest.approx(0.0)

    def test_parse_output_handles_none_timestamps(self):
        output = {"text": "hi", "chunks": [{"text": "hi", "timestamp": (None, None)}]}
        model = self._model_with_mock_pipe(output)
        result = model._parse_output(output, language="en", source="")
        assert result.segments[0].start == pytest.approx(0.0)
        assert result.segments[0].end == pytest.approx(0.0)

    def test_transcribe_sequential_calls_pipeline_correctly(self):
        pipe_output = _fake_pipeline_output(("test",))
        with patch("src.models.whisper.hf_pipeline") as mock_hf:
            mock_pipe = MagicMock()
            mock_pipe.return_value = pipe_output
            mock_pipe.feature_extractor = MagicMock()
            mock_pipe.tokenizer = MagicMock()
            mock_pipe.model = MagicMock()
            mock_hf.return_value = mock_pipe

            model = WhisperModel(model_name="openai/whisper-tiny", device="cpu")
            # patch detect_language so no forward pass needed
            model.detect_language = MagicMock(return_value="vi")

            waveform = _sine(3.0)
            result = model.transcribe_sequential(waveform, SR, language="vi", source="s")

        call_kwargs = mock_pipe.call_args
        assert call_kwargs.kwargs["return_timestamps"] is True
        assert "chunk_length_s" not in call_kwargs.kwargs
        gk = call_kwargs.kwargs["generate_kwargs"]
        assert gk["language"] == "vi"
        assert gk["condition_on_prev_tokens"] is True
        assert isinstance(result, TranscriptionResult)

    def test_transcribe_chunked_calls_pipeline_correctly(self):
        pipe_output = _fake_pipeline_output(("test",))
        with patch("src.models.whisper.hf_pipeline") as mock_hf:
            mock_pipe = MagicMock()
            mock_pipe.return_value = pipe_output
            mock_pipe.feature_extractor = MagicMock()
            mock_pipe.tokenizer = MagicMock()
            mock_pipe.model = MagicMock()
            mock_hf.return_value = mock_pipe

            model = WhisperModel(model_name="openai/whisper-tiny", device="cpu")
            model.detect_language = MagicMock(return_value="en")

            waveform = _sine(10.0)
            result = model.transcribe_chunked(
                waveform, SR, language="en", chunk_length_s=5.0,
                stride_length_s=1.0, batch_size=2, source="s"
            )

        call_kwargs = mock_pipe.call_args
        assert call_kwargs.kwargs["chunk_length_s"] == 5.0
        assert call_kwargs.kwargs["stride_length_s"] == 1.0
        assert call_kwargs.kwargs["batch_size"] == 2
        assert call_kwargs.kwargs["return_timestamps"] is True
        assert "condition_on_prev_tokens" not in call_kwargs.kwargs.get("generate_kwargs", {})
        assert isinstance(result, TranscriptionResult)


class TestTranscribeFunction:
    def _patch_model(self, result: TranscriptionResult):
        """Patch WhisperModel so transcribe() returns a fixed result."""
        mock_model = MagicMock(spec=WhisperModel)
        mock_model.transcribe_sequential.return_value = result
        mock_model.transcribe_chunked.return_value = result
        return patch("src.inference.transcribe.WhisperModel", return_value=mock_model)

    def _dummy_result(self) -> TranscriptionResult:
        return TranscriptionResult(
            text="xin chào",
            segments=[TranscriptionSegment(text="xin chào", start=0.0, end=2.0, language="vi")],
            language="vi",
            source="test",
            duration=2.0,
        )

    def test_sequential_mode_is_default(self, tmp_path):
        import soundfile as sf

        wav = tmp_path / "audio.wav"
        sf.write(str(wav), _sine(3.0), SR)

        result = self._dummy_result()
        with self._patch_model(result) as mock_cls:
            out = transcribe(str(wav), TranscribeConfig(mode="sequential"))

        mock_instance = mock_cls.return_value
        mock_instance.transcribe_sequential.assert_called_once()
        mock_instance.transcribe_chunked.assert_not_called()
        assert out.text == "xin chào"

    def test_chunked_mode_routes_to_chunked(self, tmp_path):
        import soundfile as sf

        wav = tmp_path / "audio.wav"
        sf.write(str(wav), _sine(3.0), SR)

        result = self._dummy_result()
        with self._patch_model(result) as mock_cls:
            out = transcribe(str(wav), TranscribeConfig(mode="chunked"))

        mock_instance = mock_cls.return_value
        mock_instance.transcribe_chunked.assert_called_once()
        mock_instance.transcribe_sequential.assert_not_called()

    def test_default_config(self, tmp_path):
        import soundfile as sf

        wav = tmp_path / "audio.wav"
        sf.write(str(wav), _sine(3.0), SR)

        result = self._dummy_result()
        with self._patch_model(result):
            out = transcribe(str(wav))  # no config → defaults

        assert out.language == "vi"


# ---------------------------------------------------------------------------
# Integration tests (real model, CUDA required)
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_integration_sequential(tmp_path):
    """Smoke test: sequential mode transcribes a real 20-second file."""
    import soundfile as sf

    wav = "data/test/20s_test.wav"
    config = TranscribeConfig(
        model_name="openai/whisper-large-v3",
        language="vi",
        mode="sequential",
    )
    result = transcribe(wav, config)

    assert isinstance(result.text, str) and len(result.text) > 0
    assert len(result.segments) > 0
    assert result.duration > 0
    assert all(s.end >= s.start for s in result.segments)


@pytest.mark.integration
def test_integration_chunked(tmp_path):
    """Smoke test: chunked mode transcribes a real 20-second file."""
    wav = "data/test/20s_test.wav"
    config = TranscribeConfig(
        model_name="openai/whisper-large-v3",
        language="vi",
        mode="chunked",
        chunk_length_s=10.0,
        stride_length_s=2.0,
        batch_size=2,
    )
    result = transcribe(wav, config)

    assert isinstance(result.text, str) and len(result.text) > 0
    assert len(result.segments) > 0
    assert result.duration > 0
