import io
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Union
from urllib.parse import urlparse

import numpy as np
import soundfile as sf


@dataclass
class AudioData:
    waveform: np.ndarray  # float32, shape (samples,) mono or (channels, samples) stereo
    sample_rate: int
    source: str

    @property
    def duration(self) -> float:
        return self.waveform.shape[-1] / self.sample_rate

    @property
    def n_channels(self) -> int:
        return 1 if self.waveform.ndim == 1 else self.waveform.shape[0]

    @property
    def n_samples(self) -> int:
        return self.waveform.shape[-1]


def load(
    source: Union[str, Path, bytes, io.BytesIO, np.ndarray],
    sample_rate: int = 16000,
) -> AudioData:
    """Load audio from any supported source into a float32 AudioData object.

    Supported sources:
      - Local file path (str or Path): WAV, FLAC, MP3, MP4, M4A, OGG, OPUS, …
      - YouTube URL
      - HTTP/HTTPS URL pointing to an audio file
      - Raw bytes or BytesIO
      - NumPy array (assumed float32; sample_rate arg used as its SR)
    """
    if isinstance(source, np.ndarray):
        return AudioData(
            waveform=source.astype(np.float32),
            sample_rate=sample_rate,
            source="array",
        )

    if isinstance(source, (bytes, io.BytesIO)):
        return _from_bytes(source)

    source_str = str(source)

    if _is_youtube(source_str):
        return _from_youtube(source_str)

    if source_str.startswith(("http://", "https://")):
        return _from_http(source_str)

    return _from_file(Path(source_str))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _is_youtube(url: str) -> bool:
    host = urlparse(url).netloc
    return host in {"www.youtube.com", "youtube.com", "youtu.be", "m.youtube.com"}


def _squeeze_mono(arr: np.ndarray) -> np.ndarray:
    """Squeeze (1, N) → (N,); leave (C, N) with C>1 unchanged."""
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr.squeeze(0)
    return arr


def _from_file(path: Path) -> AudioData:
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    # soundfile handles WAV/FLAC natively and is fastest for those
    if path.suffix.lower() in {".wav", ".flac"}:
        try:
            waveform, sr = sf.read(str(path), dtype="float32", always_2d=True)
            # soundfile: (samples, channels) → (channels, samples), squeeze if mono
            return AudioData(
                waveform=_squeeze_mono(waveform.T),
                sample_rate=sr,
                source=str(path),
            )
        except Exception:
            pass  # fall through

    # torchaudio covers MP3, M4A, OGG, OPUS, MP4 audio tracks, etc.
    # It ships with its own codec support — no system ffmpeg needed.
    try:
        return _torchaudio_decode(str(path), source_label=str(path))
    except Exception:
        pass

    # Last resort: system ffmpeg (if available)
    return _ffmpeg_decode(str(path), source_label=str(path))


def _from_bytes(data: Union[bytes, io.BytesIO]) -> AudioData:
    if isinstance(data, io.BytesIO):
        data = data.read()

    # Try soundfile in-memory (WAV/FLAC)
    try:
        waveform, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return AudioData(waveform=_squeeze_mono(waveform.T), sample_rate=sr, source="bytes")
    except Exception:
        pass

    # Write to temp file and try torchaudio / ffmpeg
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return _torchaudio_decode(tmp_path, source_label="bytes")
    except Exception:
        pass
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        return _ffmpeg_decode(tmp_path, source_label="bytes")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _from_http(url: str) -> AudioData:
    import requests

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    audio = _from_bytes(resp.content)
    audio.source = url
    return audio


def _from_youtube(url: str) -> AudioData:
    import yt_dlp

    with tempfile.TemporaryDirectory() as tmpdir:
        out_template = str(Path(tmpdir) / "audio.%(ext)s")
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": out_template,
            "postprocessors": [
                {"key": "FFmpegExtractAudio", "preferredcodec": "wav"}
            ],
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get("title", url)

        wav_path = Path(tmpdir) / "audio.wav"
        if not wav_path.exists():
            files = list(Path(tmpdir).iterdir())
            if not files:
                raise RuntimeError(f"yt-dlp produced no output for: {url}")
            wav_path = files[0]

        audio = _from_file(wav_path)

    audio.source = f"youtube:{title}"
    return audio


def _torchaudio_decode(input_path: str, source_label: str) -> AudioData:
    """Decode audio via torchaudio (bundled codec support, no system ffmpeg needed)."""
    import torchaudio

    waveform, sr = torchaudio.load(input_path)  # returns (channels, samples) float32 tensor
    arr = waveform.numpy()
    return AudioData(waveform=_squeeze_mono(arr), sample_rate=sr, source=source_label)


def _ffmpeg_decode(input_path: str, source_label: str) -> AudioData:
    """Decode via system ffmpeg as last resort."""
    result = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", input_path, "-f", "wav", "-"],
        capture_output=True,
        check=True,
    )
    waveform, sr = sf.read(io.BytesIO(result.stdout), dtype="float32", always_2d=True)
    return AudioData(waveform=_squeeze_mono(waveform.T), sample_rate=sr, source=source_label)
