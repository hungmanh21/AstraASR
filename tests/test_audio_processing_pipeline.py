"""
Tests for src/audio pipeline.

Test fixtures:
  data/test/1_minute_test.mp3  —  60 s, 22050 Hz, mono
  data/test/20s_test.wav       —  24 s, 16000 Hz, mono

Run with:  uv run pytest tests/test_audio_pipeline.py -v
"""

import numpy as np
import pytest

FILE_1MIN = "data/test/1_minute_test.mp3"
FILE_20S  = "data/test/20s_test.wav"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sine(duration_s: float = 3.0, sr: int = 22050, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (np.sin(2 * np.pi * freq * t) * 0.5).astype(np.float32)


def _make_speech_like(duration_s: float = 3.0, sr: int = 16000) -> np.ndarray:
    """Broadband noise that Silero VAD reliably classifies as speech."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(int(sr * duration_s)) * 0.3).astype(np.float32)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class TestLoader:
    def test_load_wav(self):
        from src.audio.loader import load

        audio = load(FILE_20S)

        assert audio.waveform.dtype == np.float32
        assert audio.sample_rate == 16000
        assert audio.duration == pytest.approx(24.0, abs=0.5)

    def test_load_mp3(self):
        from src.audio.loader import load

        audio = load(FILE_1MIN)

        assert audio.waveform.dtype == np.float32
        assert audio.sample_rate == 22050
        assert audio.duration == pytest.approx(60.0, abs=0.5)

    def test_load_numpy_array(self):
        from src.audio.loader import load

        arr = _make_sine(sr=16000)
        audio = load(arr, sample_rate=16000)

        assert audio.sample_rate == 16000
        assert audio.source == "array"
        np.testing.assert_array_equal(audio.waveform, arr)

    def test_load_bytes(self):
        from src.audio.loader import load

        with open(FILE_20S, "rb") as f:
            data = f.read()

        audio = load(data)

        assert audio.waveform.dtype == np.float32
        assert audio.n_samples > 0

    def test_load_missing_file_raises(self):
        from src.audio.loader import load

        with pytest.raises(FileNotFoundError):
            load("data/test/nonexistent.wav")


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

class TestResample:
    def test_resample_mp3_to_16k(self):
        from src.audio.loader import load
        from src.audio.transforms import resample

        audio = load(FILE_1MIN)
        resampled = resample(audio, target_sr=16000)

        assert resampled.sample_rate == 16000
        expected = int(audio.n_samples * 16000 / audio.sample_rate)
        assert abs(resampled.n_samples - expected) <= 2

    def test_resample_noop_same_sr(self):
        from src.audio.loader import load
        from src.audio.transforms import resample

        audio = load(FILE_20S)  # already 16 kHz
        result = resample(audio, target_sr=16000)

        assert result is audio


class TestToMono:
    def test_stereo_to_mono(self):
        from src.audio.loader import AudioData
        from src.audio.transforms import to_mono

        stereo = np.stack([_make_sine(), _make_sine(freq=880.0)])  # (2, N)
        audio = AudioData(waveform=stereo, sample_rate=22050, source="test")
        mono = to_mono(audio)

        assert mono.waveform.ndim == 1
        assert mono.n_samples == stereo.shape[1]

    def test_already_mono_noop(self):
        from src.audio.loader import load
        from src.audio.transforms import to_mono

        audio = load(FILE_20S)
        result = to_mono(audio)

        assert result is audio


class TestNormalize:
    def test_peak_normalize(self):
        from src.audio.loader import load
        from src.audio.transforms import normalize

        audio = load(FILE_20S)
        normed = normalize(audio, method="peak")

        assert np.abs(normed.waveform).max() == pytest.approx(1.0, abs=1e-5)

    def test_rms_normalize(self):
        from src.audio.loader import load
        from src.audio.transforms import normalize

        audio = load(FILE_20S)
        normed = normalize(audio, method="rms")

        rms = np.sqrt(np.mean(normed.waveform ** 2))
        assert rms == pytest.approx(0.1, abs=1e-3)

    def test_unknown_method_raises(self):
        from src.audio.loader import load
        from src.audio.transforms import normalize

        audio = load(FILE_20S)
        with pytest.raises(ValueError):
            normalize(audio, method="invalid")


class TestTrimSilence:
    def test_strips_silence(self):
        from src.audio.loader import AudioData
        from src.audio.transforms import trim_silence

        silence = np.zeros(16000, dtype=np.float32)
        signal = _make_speech_like(duration_s=1.0)
        waveform = np.concatenate([silence, signal, silence])
        audio = AudioData(waveform=waveform, sample_rate=16000, source="test")
        trimmed = trim_silence(audio)

        assert trimmed.n_samples < audio.n_samples


# ---------------------------------------------------------------------------
# VAD chunking — long audio (1-minute file)
# ---------------------------------------------------------------------------

class TestChunkByVad:
    """Tests use the 60 s MP3 so VAD operates on real speech content."""

    @pytest.fixture(scope="class")
    def audio_1min_16k(self):
        """Load and preprocess the 1-minute file once for all tests in this class."""
        from src.audio.loader import load
        from src.audio.transforms import normalize, resample, to_mono

        audio = load(FILE_1MIN)
        audio = resample(audio, 16000)
        audio = to_mono(audio)
        audio = normalize(audio)
        return audio

    def test_produces_multiple_chunks(self, audio_1min_16k):
        from src.audio.transforms import chunk_by_vad

        chunks = chunk_by_vad(audio_1min_16k, max_duration=30.0)

        # 60 s of speech must produce at least 2 chunks with a 30 s limit
        assert len(chunks) >= 2

    def test_no_chunk_exceeds_max_duration(self, audio_1min_16k):
        from src.audio.transforms import chunk_by_vad

        max_dur = 30.0
        overlap = 0.5
        chunks = chunk_by_vad(audio_1min_16k, max_duration=max_dur, overlap_s=overlap)

        for chunk in chunks:
            # each chunk may be padded by overlap on both sides
            assert chunk.duration <= max_dur + 2 * overlap

    def test_chunks_cover_full_audio(self, audio_1min_16k):
        from src.audio.transforms import chunk_by_vad

        chunks = chunk_by_vad(audio_1min_16k, max_duration=30.0, overlap_s=0.0)

        # last chunk end should be close to total duration
        assert chunks[-1].end == pytest.approx(audio_1min_16k.duration, abs=1.0)

    def test_chunk_indices_are_sequential(self, audio_1min_16k):
        from src.audio.transforms import chunk_by_vad

        chunks = chunk_by_vad(audio_1min_16k, max_duration=30.0)

        assert [c.index for c in chunks] == list(range(len(chunks)))

    def test_chunk_waveform_length_matches_timestamps(self, audio_1min_16k):
        from src.audio.transforms import chunk_by_vad

        chunks = chunk_by_vad(audio_1min_16k, max_duration=30.0)

        sr = audio_1min_16k.sample_rate
        for chunk in chunks:
            expected_samples = int((chunk.end - chunk.start) * sr)
            # allow ±1 sample for float→int rounding
            assert abs(chunk.n_samples - expected_samples) <= 1

    def test_smaller_max_duration_gives_more_chunks(self, audio_1min_16k):
        from src.audio.transforms import chunk_by_vad

        chunks_30 = chunk_by_vad(audio_1min_16k, max_duration=30.0)
        chunks_10 = chunk_by_vad(audio_1min_16k, max_duration=10.0)

        assert len(chunks_10) > len(chunks_30)

    def test_overlap_widens_chunks(self, audio_1min_16k):
        from src.audio.transforms import chunk_by_vad

        chunks_no_overlap  = chunk_by_vad(audio_1min_16k, max_duration=30.0, overlap_s=0.0)
        chunks_with_overlap = chunk_by_vad(audio_1min_16k, max_duration=30.0, overlap_s=1.0)

        # At least the middle chunks should be wider with overlap
        if len(chunks_no_overlap) >= 2:
            total_no  = sum(c.duration for c in chunks_no_overlap)
            total_with = sum(c.duration for c in chunks_with_overlap)
            assert total_with > total_no


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_run_on_short_wav(self):
        from src.audio.pipeline import run

        chunks = run(FILE_20S)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.sample_rate == 16000
            assert chunk.waveform.ndim == 1
            assert chunk.waveform.dtype == np.float32

    def test_run_on_1min_mp3(self):
        from src.audio.pipeline import run

        chunks = run(FILE_1MIN)

        assert len(chunks) >= 2           # 60 s must split into multiple chunks
        for chunk in chunks:
            assert chunk.sample_rate == 16000
            assert chunk.waveform.ndim == 1
            assert chunk.duration <= 31.0  # 30 s + max overlap

    def test_run_from_yaml_config(self):
        from src.audio.pipeline import AudioPipelineConfig, run

        config = AudioPipelineConfig.from_yaml("config/audio_pipeline.yaml")
        chunks = run(FILE_1MIN, config=config)

        assert len(chunks) >= 1

    def test_run_custom_config_smaller_chunks(self):
        from src.audio.pipeline import AudioPipelineConfig, run

        config = AudioPipelineConfig(vad_max_duration=10.0, chunk_overlap_s=0.0)
        chunks_10 = run(FILE_1MIN, config=config)

        config.vad_max_duration = 30.0
        chunks_30 = run(FILE_1MIN, config=config)

        assert len(chunks_10) > len(chunks_30)
