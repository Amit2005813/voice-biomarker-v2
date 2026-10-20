"""Shared utilities for the Voice Biomarker Platform v2 data loaders.

Provides audio file validation, feature caching, label encoding/decoding,
and data augmentation functions used across all loaders.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Audio file validation
# ---------------------------------------------------------------------------

_SUPPORTED_EXTENSIONS = frozenset({".wav", ".mp3", ".ogg", ".flac", ".m4a"})


class AudioValidationError(Exception):
    """Raised when an audio file fails validation checks."""


def validate_audio_file(
    filepath: Union[str, Path],
    *,
    min_duration_sec: float = 0.5,
    max_duration_sec: float = 120.0,
    expected_sr: Optional[int] = None,
    min_amplitude: float = 1e-6,
) -> Dict[str, Any]:
    """Validate an audio file and return its metadata.

    Parameters
    ----------
    filepath:
        Path to the audio file.
    min_duration_sec:
        Minimum acceptable duration in seconds.
    max_duration_sec:
        Maximum acceptable duration in seconds.
    expected_sr:
        If provided, verify the sample rate matches.
    min_amplitude:
        Minimum peak amplitude to consider the file non-silent.

    Returns
    -------
    dict
        Metadata dict with keys: ``duration_sec``, ``sample_rate``,
        ``num_channels``, ``num_frames``, ``file_size_bytes``.

    Raises
    ------
    AudioValidationError
        If the file is missing, unreadable, or fails any check.
    """
    path = Path(filepath)

    if not path.exists():
        raise AudioValidationError(f"Audio file not found: {path}")

    ext = path.suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        raise AudioValidationError(
            f"Unsupported audio format '{ext}'. "
            f"Supported: {sorted(_SUPPORTED_EXTENSIONS)}"
        )

    file_size = path.stat().st_size
    if file_size == 0:
        raise AudioValidationError(f"Audio file is empty: {path}")

    metadata: Dict[str, Any] = {"file_size_bytes": file_size, "path": str(path)}

    if ext == ".wav":
        metadata.update(_validate_wav(path, expected_sr, min_amplitude))
    else:
        # For non-WAV formats we rely on the file existing and being non-empty.
        # Full validation requires librosa/soundfile at the call-site.
        metadata["format"] = ext.lstrip(".")
        metadata["duration_sec"] = None
        metadata["sample_rate"] = None

    duration = metadata.get("duration_sec")
    if duration is not None:
        if duration < min_duration_sec:
            raise AudioValidationError(
                f"Audio too short ({duration:.2f}s < {min_duration_sec}s): {path}"
            )
        if duration > max_duration_sec:
            raise AudioValidationError(
                f"Audio too long ({duration:.2f}s > {max_duration_sec}s): {path}"
            )

    if expected_sr is not None and metadata.get("sample_rate") is not None:
        if metadata["sample_rate"] != expected_sr:
            raise AudioValidationError(
                f"Sample rate mismatch: expected {expected_sr}, "
                f"got {metadata['sample_rate']}: {path}"
            )

    return metadata


def _validate_wav(
    path: Path,
    expected_sr: Optional[int],
    min_amplitude: float,
) -> Dict[str, Any]:
    """Validate a WAV file and return metadata."""
    try:
        with wave.open(str(path), "rb") as wf:
            num_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            num_frames = wf.getnframes()
            duration = num_frames / sample_rate if sample_rate else 0.0

            if expected_sr is not None and sample_rate != expected_sr:
                raise AudioValidationError(
                    f"Sample rate mismatch: expected {expected_sr}, "
                    f"got {sample_rate}: {path}"
                )

            # Quick amplitude check on first chunk
            raw = wf.readframes(min(num_frames, sample_rate))  # 1 s max
            if sample_width == 2:
                samples = np.frombuffer(raw, dtype=np.int16)
                peak = np.max(np.abs(samples.astype(np.float64))) / 32768.0
            elif sample_width == 4:
                samples = np.frombuffer(raw, dtype=np.int32)
                peak = np.max(np.abs(samples.astype(np.float64))) / 2147483648.0
            else:
                peak = 1.0  # skip check for uncommon widths

            if peak < min_amplitude:
                raise AudioValidationError(
                    f"Audio appears silent (peak={peak:.2e}): {path}"
                )

            return {
                "duration_sec": duration,
                "sample_rate": sample_rate,
                "num_channels": num_channels,
                "num_frames": num_frames,
                "sample_width": sample_width,
                "format": "wav",
            }
    except AudioValidationError:
        raise
    except Exception as exc:
        raise AudioValidationError(f"Cannot read WAV file {path}: {exc}") from exc


def check_audio_directory(
    directory: Union[str, Path],
    *,
    recursive: bool = True,
    min_files: int = 1,
) -> Tuple[List[Path], List[str]]:
    """Scan a directory for audio files.

    Returns
    -------
    (valid_paths, errors)
        Tuple of valid audio paths and error message strings.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {dir_path}")

    valid: List[Path] = []
    errors: List[str] = []
    pattern = "**/*" if recursive else "*"

    for fpath in dir_path.glob(pattern):
        if not fpath.is_file():
            continue
        if fpath.suffix.lower() in _SUPPORTED_EXTENSIONS:
            valid.append(fpath)

    if len(valid) < min_files:
        errors.append(
            f"Found {len(valid)} audio files in {dir_path}, expected >= {min_files}"
        )

    return valid, errors


# ---------------------------------------------------------------------------
# Feature caching
# ---------------------------------------------------------------------------


def _cache_key(label: str, **kwargs: Any) -> str:
    """Deterministic cache key from label + parameters."""
    raw = json.dumps({"label": label, **kwargs}, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class FeatureCache:
    """Simple JSON-based feature cache on disk."""

    def __init__(self, cache_dir: Union[str, Path]) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.cache_dir / "index.json"
        self._index: Dict[str, str] = {}
        self._load_index()

    # -- internal helpers ---------------------------------------------------

    def _load_index(self) -> None:
        if self._index_path.exists():
            with open(self._index_path, "r", encoding="utf-8") as fh:
                self._index = json.load(fh)

    def _save_index(self) -> None:
        with open(self._index_path, "w", encoding="utf-8") as fh:
            json.dump(self._index, fh, indent=2)

    # -- public API ---------------------------------------------------------

    def get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve cached features by key, or *None* on miss."""
        if key not in self._index:
            return None
        fpath = self.cache_dir / self._index[key]
        if not fpath.exists():
            return None
        return np.load(str(fpath))

    def put(self, key: str, features: np.ndarray) -> None:
        """Store features to disk and update the index."""
        fname = f"{key}.npy"
        fpath = self.cache_dir / fname
        np.save(str(fpath), features)
        self._index[key] = fname
        self._save_index()

    def has(self, key: str) -> bool:
        return key in self._index and (self.cache_dir / self._index[key]).exists()

    def clear(self) -> None:
        """Remove all cached files."""
        for fname in self._index.values():
            fpath = self.cache_dir / fname
            if fpath.exists():
                fpath.unlink()
        self._index.clear()
        self._save_index()

    def __len__(self) -> int:
        return len(self._index)


# ---------------------------------------------------------------------------
# Label encoding / decoding
# ---------------------------------------------------------------------------


class LabelEncoder:
    """Simple mapping-based label encoder.

    Encodes arbitrary labels to contiguous integers and provides
    round-trip decoding.

    Examples
    --------
    >>> enc = LabelEncoder()
    >>> enc.fit(["healthy", "parkinson", "healthy", "depression"])
    >>> enc.encode("healthy")
    0
    >>> enc.decode(2)
    'parkinson'
    """

    def __init__(self) -> None:
        self._class_to_idx: Dict[str, int] = {}
        self._idx_to_class: Dict[int, str] = {}

    @property
    def classes(self) -> List[str]:
        """Return sorted list of known class labels."""
        return sorted(self._class_to_idx.keys())

    @property
    def num_classes(self) -> int:
        return len(self._class_to_idx)

    def fit(self, labels: Sequence[str]) -> "LabelEncoder":
        """Learn the mapping from *labels*."""
        unique = sorted(set(labels))
        self._class_to_idx = {c: i for i, c in enumerate(unique)}
        self._idx_to_class = {i: c for c, i in self._class_to_idx.items()}
        return self

    def encode(self, label: str) -> int:
        """Encode a single label to its integer index."""
        if label not in self._class_to_idx:
            raise KeyError(f"Unknown label '{label}'. Call fit() first.")
        return self._class_to_idx[label]

    def encode_many(self, labels: Sequence[str]) -> np.ndarray:
        """Encode a sequence of labels to a NumPy int array."""
        return np.array([self.encode(l) for l in labels], dtype=np.int64)

    def decode(self, idx: int) -> str:
        """Decode an integer index back to its label."""
        if idx not in self._idx_to_class:
            raise KeyError(f"Unknown index {idx}.")
        return self._idx_to_class[idx]

    def decode_many(self, indices: Sequence[int]) -> List[str]:
        """Decode a sequence of indices."""
        return [self.decode(i) for i in indices]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the encoder to a JSON-compatible dict."""
        return {
            "class_to_idx": self._class_to_idx,
            "idx_to_class": {str(k): v for k, v in self._idx_to_class.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LabelEncoder":
        """Restore an encoder from a serialized dict."""
        enc = cls()
        enc._class_to_idx = data["class_to_idx"]
        enc._idx_to_class = {int(k): v for k, v in data["idx_to_class"].items()}
        return enc

    def save(self, filepath: Union[str, Path]) -> None:
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "LabelEncoder":
        with open(filepath, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

_rng = np.random.default_rng(seed=42)


def time_stretch(
    audio: np.ndarray,
    rate: float = 1.0,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Simple resample-based time stretch (no pitch change approximation).

    Parameters
    ----------
    audio:
        1-D audio array.
    rate:
        Stretch factor. ``1.0`` = no change, ``> 1`` = slower, ``< 1`` = faster.
    rng:
        Optional NumPy random generator.

    Returns
    -------
    np.ndarray
        Stretched audio array (may differ in length).
    """
    if rate == 1.0:
        return audio.copy()
    indices = np.linspace(0, len(audio) - 1, int(len(audio) * rate))
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def pitch_shift(
    audio: np.ndarray,
    sample_rate: int,
    n_semitones: float = 0.0,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Resampling-based pitch shift (shifts pitch and speed together).

    Parameters
    ----------
    audio:
        1-D audio array.
    sample_rate:
        Audio sample rate in Hz.
    n_semitones:
        Number of semitones to shift (positive = higher pitch).
    rng:
        Optional NumPy random generator.

    Returns
    -------
    np.ndarray
        Pitch-shifted audio.
    """
    if n_semitones == 0.0:
        return audio.copy()
    factor = 2 ** (n_semitones / 12.0)
    indices = np.linspace(0, len(audio) - 1, int(len(audio) / factor))
    shifted = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
    return shifted


def add_noise(
    audio: np.ndarray,
    noise_level: float = 0.005,
    *,
    noise_type: str = "gaussian",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add noise to an audio signal.

    Parameters
    ----------
    audio:
        1-D audio array.
    noise_level:
        Standard deviation of additive noise (relative to peak amplitude).
    noise_type:
        ``"gaussian"`` or ``"uniform"``.
    rng:
        Optional NumPy random generator.

    Returns
    -------
    np.ndarray
        Noisy audio.
    """
    _rng2 = rng or _rng
    peak = np.max(np.abs(audio)) if np.any(audio) else 1.0
    scale = noise_level * peak

    if noise_type == "gaussian":
        noise = _rng2.normal(0, scale, size=len(audio)).astype(np.float32)
    elif noise_type == "uniform":
        noise = _rng2.uniform(-scale, scale, size=len(audio)).astype(np.float32)
    else:
        raise ValueError(f"Unknown noise_type '{noise_type}'")

    return np.clip(audio + noise, -1.0, 1.0).astype(np.float32)


def volume_normalize(audio: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Peak-normalize audio to *target_peak*."""
    peak = np.max(np.abs(audio))
    if peak < 1e-8:
        return audio.copy()
    return (audio / peak * target_peak).astype(np.float32)


def random_crop(
    audio: np.ndarray,
    num_samples: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> Optional[np.ndarray]:
    """Randomly crop *audio* to *num_samples*.

    Returns *None* if the audio is shorter than *num_samples*.
    """
    _rng2 = rng or _rng
    if len(audio) < num_samples:
        return None
    start = int(_rng2.integers(0, len(audio) - num_samples))
    return audio[start : start + num_samples].copy()


def random_augment(
    audio: np.ndarray,
    sample_rate: int = 16000,
    *,
    prob_stretch: float = 0.3,
    prob_pitch: float = 0.3,
    prob_noise: float = 0.5,
    prob_crop: float = 0.0,
    target_samples: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply a random chain of augmentations to *audio*.

    Useful for training-data augmentation pipelines.

    Parameters
    ----------
    audio:
        1-D input audio array.
    sample_rate:
        Sample rate in Hz.
    prob_stretch:
        Probability of applying time stretching.
    prob_pitch:
        Probability of applying pitch shifting.
    prob_noise:
        Probability of adding additive noise.
    prob_crop:
        Probability of random cropping (requires *target_samples*).
    target_samples:
        Target length for random cropping.
    rng:
        Optional NumPy random generator.

    Returns
    -------
    np.ndarray
        Augmented audio.
    """
    _rng2 = rng or _rng
    result = audio.copy()

    if _rng2.random() < prob_stretch and len(result) > 100:
        rate = float(_rng2.uniform(0.9, 1.1))
        result = time_stretch(result, rate=rate, rng=_rng2)

    if _rng2.random() < prob_pitch:
        shift = float(_rng2.uniform(-2.0, 2.0))
        result = pitch_shift(result, sample_rate, n_semitones=shift, rng=_rng2)

    if _rng2.random() < prob_noise:
        level = float(_rng2.uniform(0.001, 0.01))
        result = add_noise(result, noise_level=level, rng=_rng2)

    if (
        prob_crop > 0
        and target_samples is not None
        and _rng2.random() < prob_crop
    ):
        cropped = random_crop(result, target_samples, rng=_rng2)
        if cropped is not None:
            result = cropped

    return result


# ---------------------------------------------------------------------------
# Stratified split helper
# ---------------------------------------------------------------------------


def stratified_split(
    labels: Sequence[int],
    *,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Return stratified train/val/test index lists.

    Parameters
    ----------
    labels:
        Integer-encoded labels for every sample.
    train_ratio, val_ratio, test_ratio:
        Proportions (must sum to ~1.0).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict
        ``{"train": [...], "val": [...], "test": [...]}`` of integer indices.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    rng = np.random.default_rng(seed)
    indices_by_class: Dict[int, List[int]] = {}
    for i, lbl in enumerate(labels):
        indices_by_class.setdefault(lbl, []).append(i)

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []

    for cls_indices in indices_by_class.values():
        perm = rng.permutation(cls_indices).tolist()
        n = len(perm)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(perm[:n_train])
        val_idx.extend(perm[n_train : n_train + n_val])
        test_idx.extend(perm[n_train + n_val :])

    return {"train": train_idx, "val": val_idx, "test": test_idx}


# ---------------------------------------------------------------------------
# JSON I/O helpers
# ---------------------------------------------------------------------------


def save_json(data: Any, filepath: Union[str, Path], *, indent: int = 2) -> None:
    """Write *data* to a JSON file."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False)


def load_json(filepath: Union[str, Path]) -> Any:
    """Read and return data from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Audio synthesis helpers (used by synthetic loader)
# ---------------------------------------------------------------------------


def synthesize_formant_signal(
    duration_sec: float,
    sample_rate: int,
    fundamental_freq: float,
    formant_freqs: Sequence[float],
    formant_bws: Sequence[float],
    *,
    jitter_pct: float = 0.5,
    shimmer_pct: float = 3.0,
    noise_level: float = 0.002,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Synthesize a vowel-like signal with realistic formant structure.

    Parameters
    ----------
    duration_sec:
        Duration of the signal in seconds.
    sample_rate:
        Sample rate in Hz.
    fundamental_freq:
        Fundamental frequency (F0) in Hz.
    formant_freqs:
        Center frequencies of the formants (e.g. [700, 1200, 2600]).
    formant_bws:
        Bandwidths for each formant in Hz.
    jitter_pct:
        Cycle-to-cycle frequency perturbation (%).
    shimmer_pct:
        Cycle-to-cycle amplitude perturbation (%).
    noise_level:
        Additive broadband noise level.
    rng:
        NumPy random generator.

    Returns
    -------
    np.ndarray
        Synthesized audio waveform in [-1, 1].
    """
    _rng3 = rng or _rng
    n_samples = int(duration_sec * sample_rate)
    output = np.zeros(n_samples, dtype=np.float64)

    # Glottal pulse train with jitter and shimmer
    period_samples = sample_rate / fundamental_freq
    t = 0.0
    amp = 1.0
    while int(t) < n_samples:
        # Jitter: perturb the period
        jittered_period = period_samples * (1.0 + _rng3.normal(0, jitter_pct / 100.0))
        # Shimmer: perturb the amplitude
        amp = 1.0 + _rng3.normal(0, shimmer_pct / 100.0)

        idx = int(t)
        if idx < n_samples:
            output[idx] += amp
        t += jittered_period

    # Apply formant filters (cascade of second-order resonators)
    for fc, bw in zip(formant_freqs, formant_bws):
        output = _apply_resonator(output, fc, bw, sample_rate)

    # Add breath noise
    output += _rng3.normal(0, noise_level, size=n_samples)

    # Normalize
    peak = np.max(np.abs(output))
    if peak > 0:
        output = output / peak * 0.9

    return output.astype(np.float32)


def _apply_resonator(
    signal: np.ndarray,
    center_freq: float,
    bandwidth: float,
    sample_rate: int,
) -> np.ndarray:
    """Apply a second-order resonator (formant filter) to a signal."""
    r = np.exp(-np.pi * bandwidth / sample_rate)
    theta = 2 * np.pi * center_freq / sample_rate
    b0 = 1.0 - r
    a1 = -2 * r * np.cos(theta)
    a2 = r * r

    output = np.zeros_like(signal)
    x1 = x2 = y1 = y2 = 0.0

    for i in range(len(signal)):
        y0 = b0 * signal[i] - a1 * x1 - a2 * x2
        x2, x1 = x1, y0
        output[i] = y0

    return output


def generate_noise_burst(
    duration_sec: float,
    sample_rate: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate a shaped noise burst (simulates fricatives/aspiration)."""
    _rng4 = rng or _rng
    n_samples = int(duration_sec * sample_rate)
    noise = _rng4.normal(0, 1, size=n_samples).astype(np.float64)

    # Simple low-pass at 4 kHz to shape the noise
    cutoff = 4000.0 / sample_rate
    _b, _a = _butterworth_lp(4, cutoff)
    noise = _filter_iir(noise, _b, _a)

    peak = np.max(np.abs(noise))
    if peak > 0:
        noise = noise / peak * 0.5
    return noise.astype(np.float32)


def _butterworth_lp(order: int, cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
    """Simplified Butterworth low-pass coefficients via bilinear transform."""
    # Analog prototype poles
    k = np.arange(1, order + 1)
    angles = np.pi * (2 * k + order) / (2 * order)
    s_poles = np.exp(1j * angles)
    s_poles = np.concatenate([s_poles, np.conj(s_poles)])

    # Bilinear transform (pre-warp)
    wc = np.tan(np.pi * cutoff / 2)
    z_poles = (1 + s_poles * wc / (2)) / (1 - s_poles * wc / (2))
    z_poles = np.real(z_poles)  # approximate

    # Expand polynomial
    b_poly = np.poly(z_poles)
    a_poly = np.poly(z_poles * 0.95)  # slight damping for stability

    # Normalize
    b_poly = b_poly / np.sum(b_poly)
    a_poly = a_poly / a_poly[0]

    return b_poly.astype(np.float64), a_poly.astype(np.float64)


def _filter_iir(signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Apply a direct-form I IIR filter."""
    n = len(signal)
    output = np.zeros(n, dtype=np.float64)
    x_mem = np.zeros(len(b), dtype=np.float64)
    y_mem = np.zeros(len(a) - 1, dtype=np.float64)

    for i in range(n):
        x_mem = np.roll(x_mem, 1)
        x_mem[0] = signal[i]
        y_val = np.dot(b, x_mem) - np.dot(a[1:], y_mem)
        y_mem = np.roll(y_mem, 1)
        y_mem[0] = y_val
        output[i] = y_val

    return output
