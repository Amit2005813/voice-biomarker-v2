"""Enhanced audio preprocessing for Voice Biomarker Platform v2."""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import librosa
except ImportError:
    raise ImportError("librosa is required: pip install librosa")

try:
    import noisereduce as nr
except ImportError:
    nr = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma")


@dataclass
class AudioMetadata:
    filepath: str
    duration: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    samples: int = 0
    dtype: str = ""
    format_ext: str = ""


class AudioPreprocessor:
    """Full-featured audio preprocessing pipeline.

    Supports loading, noise reduction, trimming, normalization, resampling,
    data augmentation, and quality scoring of audio files.

    Example::

        ap = AudioPreprocessor(target_sr=22050)
        audio, sr = ap.preprocess("recording.wav")
    """

    def __init__(
        self,
        target_sr: int = 22050,
        peak_target: float = 0.95,
        trim_db: float = 25.0,
        noise_reduce_stationary: bool = True,
        noise_reduce_prop_decrease: float = 0.75,
        augment_prob: float = 0.5,
        time_stretch_range: Tuple[float, float] = (0.8, 1.2),
        pitch_shift_range: Tuple[float, float] = (-2.0, 2.0),
        augmentation_noise_level: float = 0.005,
    ) -> None:
        self.target_sr = target_sr
        self.peak_target = peak_target
        self.trim_db = trim_db
        self.noise_reduce_stationary = noise_reduce_stationary
        self.noise_reduce_prop_decrease = noise_reduce_prop_decrease
        self.augment_prob = augment_prob
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.augmentation_noise_level = augmentation_noise_level

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(
        self,
        filepath: Union[str, Path],
        sr: Optional[int] = None,
        mono: bool = True,
        duration: Optional[float] = None,
        offset: float = 0.0,
    ) -> Tuple[np.ndarray, int]:
        """Load an audio file using librosa.

        Args:
            filepath: Path to the audio file.
            sr: Sample rate to load at. ``None`` preserves the native rate.
            mono: Convert to mono if ``True``.
            duration: Maximum duration in seconds to load.
            offset: Start reading at this offset in seconds.

        Returns:
            Tuple of (audio_array, sample_rate).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file extension is not supported.
            RuntimeError: If librosa fails to load the file.
        """
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio format '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        effective_sr = sr if sr is not None else self.target_sr

        try:
            y, native_sr = librosa.load(
                str(path),
                sr=effective_sr,
                mono=mono,
                duration=duration,
                offset=offset,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load audio file '{filepath}': {exc}") from exc

        logger.info(
            "Loaded %s: duration=%.2fs, sr=%d, samples=%d",
            path.name,
            librosa.get_duration(y=y, sr=native_sr),
            native_sr,
            len(y),
        )
        return y, native_sr

    def _load_metadata(self, filepath: Union[str, Path]) -> AudioMetadata:
        """Collect metadata without loading full waveform when possible."""
        path = Path(filepath)
        try:
            info = librosa.get_duration(path=str(path))
            y_test, sr_test = librosa.load(str(path), sr=self.target_sr, duration=0.1)
            return AudioMetadata(
                filepath=str(path),
                duration=info,
                sample_rate=sr_test,
                channels=1,
                dtype=str(y_test.dtype),
                format_ext=path.suffix.lower(),
            )
        except Exception:
            return AudioMetadata(filepath=str(path), format_ext=path.suffix.lower())

    # ------------------------------------------------------------------
    # Noise reduction
    # ------------------------------------------------------------------

    def reduce_noise(
        self,
        y: np.ndarray,
        sr: int,
        stationary: Optional[bool] = None,
        prop_decrease: Optional[float] = None,
    ) -> np.ndarray:
        """Apply noise reduction using the noisereduce library.

        Falls back to a simple high-pass filter when noisereduce is not
        installed.

        Args:
            y: Audio signal.
            sr: Sample rate.
            stationary: Use stationary noise reduction. ``None`` uses the
                instance default.
            prop_decrease: Proportion to decrease noise by. ``None`` uses
                the instance default.

        Returns:
            Denoised audio signal.
        """
        use_stationary = stationary if stationary is not None else self.noise_reduce_stationary
        use_prop = prop_decrease if prop_decrease is not None else self.noise_reduce_prop_decrease

        if nr is not None:
            try:
                reduced = nr.reduce_noise(
                    y=y,
                    sr=sr,
                    stationary=use_stationary,
                    prop_decrease=use_prop,
                )
                logger.info("Noise reduction applied (stationary=%s)", use_stationary)
                return reduced
            except Exception as exc:
                logger.warning("noisereduce failed, falling back to median filter: %s", exc)

        return self._fallback_denoise(y)

    @staticmethod
    def _fallback_denoise(y: np.ndarray) -> np.ndarray:
        """Simple noise gate fallback when noisereduce is unavailable."""
        threshold = np.median(np.abs(y)) * 0.5
        mask = np.abs(y) > threshold
        return y * mask.astype(y.dtype)

    # ------------------------------------------------------------------
    # Trimming
    # ------------------------------------------------------------------

    def trim_silence(
        self,
        y: np.ndarray,
        sr: int,
        top_db: Optional[float] = None,
    ) -> np.ndarray:
        """Trim leading and trailing silence from the audio.

        Args:
            y: Audio signal.
            sr: Sample rate.
            top_db: Threshold in dB below peak to consider as silence.
                ``None`` uses the instance default.

        Returns:
            Trimmed audio signal.
        """
        db = top_db if top_db is not None else self.trim_db
        trimmed, _ = librosa.effects.trim(y, top_db=db)
        removed = len(y) - len(trimmed)
        logger.info("Trimmed %d samples (%.2fs of silence)", removed, removed / sr)
        return trimmed

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize(
        self,
        y: np.ndarray,
        method: str = "peak",
        target: Optional[float] = None,
    ) -> np.ndarray:
        """Normalize audio volume.

        Args:
            y: Audio signal.
            method: ``"peak"`` normalizes to peak amplitude; ``"rms"``
                normalizes to a target RMS level.
            target: Target level. Defaults to ``self.peak_target`` for peak
                normalization, ``0.1`` for RMS.

        Returns:
            Normalized audio signal.
        """
        if method == "peak":
            t = target if target is not None else self.peak_target
            peak = np.max(np.abs(y))
            if peak < 1e-8:
                logger.warning("Audio is near-silent, skipping peak normalization")
                return y
            return y * (t / peak)

        if method == "rms":
            t = target if target is not None else 0.1
            rms = np.sqrt(np.mean(y ** 2))
            if rms < 1e-8:
                logger.warning("Audio is near-silent, skipping RMS normalization")
                return y
            return y * (t / rms)

        raise ValueError(f"Unknown normalization method: {method!r}. Use 'peak' or 'rms'.")

    # ------------------------------------------------------------------
    # Resampling
    # ------------------------------------------------------------------

    def resample(self, y: np.ndarray, orig_sr: int, target_sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """Resample audio to a target sample rate.

        Args:
            y: Audio signal.
            orig_sr: Current sample rate.
            target_sr: Desired sample rate. ``None`` uses ``self.target_sr``.

        Returns:
            Tuple of (resampled_audio, new_sample_rate).
        """
        desired = target_sr if target_sr is not None else self.target_sr
        if orig_sr == desired:
            return y, orig_sr

        resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=desired)
        logger.info("Resampled from %d Hz to %d Hz", orig_sr, desired)
        return resampled, desired

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def augment(self, y: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Apply data augmentation techniques.

        Generates augmented versions of the input audio:
        - Time stretching
        - Pitch shifting
        - Adding Gaussian noise

        Augmentations are applied independently with probability
        ``self.augment_prob``.

        Args:
            y: Audio signal.
            sr: Sample rate.

        Returns:
            Dictionary mapping augmentation name to augmented audio. The
            original is always included under ``"original"``.
        """
        augmented: Dict[str, np.ndarray] = {"original": y.copy()}
        rng = np.random.default_rng()

        # Time stretch
        if rng.random() < self.augment_prob:
            rate = rng.uniform(*self.time_stretch_range)
            stretched = librosa.effects.time_stretch(y, rate=rate)
            augmented["time_stretch"] = stretched

        # Pitch shift
        if rng.random() < self.augment_prob:
            steps = rng.uniform(*self.pitch_shift_range)
            shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            augmented["pitch_shift"] = shifted

        # Additive noise
        if rng.random() < self.augment_prob:
            noise = rng.normal(0, self.augmentation_noise_level, size=y.shape).astype(y.dtype)
            noisy = y + noise
            peak = np.max(np.abs(noisy))
            if peak > 1e-6:
                noisy = noisy / peak * self.peak_target
            augmented["add_noise"] = noisy

        logger.info("Generated %d augmented variants", len(augmented) - 1)
        return augmented

    # ------------------------------------------------------------------
    # Quality scoring
    # ------------------------------------------------------------------

    def get_quality_score(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """Compute SNR-based quality metrics.

        Args:
            y: Audio signal.
            sr: Sample rate.

        Returns:
            Dictionary with keys ``snr_db``, ``rms``, ``peak``, and
            ``quality_score`` (0-1).
        """
        signal_power = float(np.mean(y ** 2))
        sorted_abs = np.sort(np.abs(y))
        n_noise = max(1, len(y) // 10)
        noise_power = float(np.mean(sorted_abs[:n_noise] ** 2))

        if noise_power < 1e-12:
            snr_db = 60.0
        else:
            snr_db = float(10.0 * np.log10(max(signal_power, 1e-12) / noise_power))

        rms = float(np.sqrt(signal_power))
        peak = float(np.max(np.abs(y)))
        quality_score = float(np.clip((snr_db + 10) / 50.0, 0.0, 1.0))

        return {
            "snr_db": round(snr_db, 2),
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "quality_score": round(quality_score, 4),
        }

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_audio(
        self,
        y: np.ndarray,
        sr: int,
        min_duration: float = 0.5,
        max_duration: float = 600.0,
        min_snr_db: float = 5.0,
    ) -> Dict[str, Any]:
        """Check whether the audio is usable for downstream analysis.

        Args:
            y: Audio signal.
            sr: Sample rate.
            min_duration: Minimum acceptable duration in seconds.
            max_duration: Maximum acceptable duration in seconds.
            min_snr_db: Minimum acceptable SNR in dB.

        Returns:
            Dictionary with ``valid`` (bool), ``issues`` (list of strings),
            and ``metrics`` (quality and duration info).
        """
        duration = len(y) / sr
        quality = self.get_quality_score(y, sr)
        issues: List[str] = []

        if duration < min_duration:
            issues.append(f"Duration {duration:.2f}s is below minimum {min_duration:.2f}s")
        if duration > max_duration:
            issues.append(f"Duration {duration:.2f}s exceeds maximum {max_duration:.2f}s")
        if quality["snr_db"] < min_snr_db:
            issues.append(f"SNR {quality['snr_db']:.1f}dB is below minimum {min_snr_db:.1f}dB")
        if quality["peak"] < 1e-4:
            issues.append("Signal amplitude is near zero")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "metrics": {
                "duration": round(duration, 3),
                "sample_rate": sr,
                **quality,
            },
        }

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def preprocess(
        self,
        filepath: Union[str, Path],
        noise_reduce: bool = True,
        normalize: bool = True,
        normalize_method: str = "peak",
    ) -> Tuple[np.ndarray, int]:
        """Run the full preprocessing pipeline.

        Pipeline steps:
            1. Load audio
            2. Noise reduction (optional)
            3. Trim silence
            4. Normalize (optional)
            5. Resample to target sample rate

        Args:
            filepath: Path to the audio file.
            noise_reduce: Whether to apply noise reduction.
            normalize: Whether to normalize volume.
            normalize_method: Normalization method (``"peak"`` or ``"rms"``).

        Returns:
            Tuple of (processed_audio, sample_rate).

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported.
            RuntimeError: If loading fails.
        """
        y, sr = self.load(filepath)
        meta = self._load_metadata(filepath)

        if noise_reduce:
            y = self.reduce_noise(y, sr)

        y = self.trim_silence(y, sr)

        if normalize:
            y = self.normalize(y, method=normalize_method)

        if sr != self.target_sr:
            y, sr = self.resample(y, sr)

        validation = self.validate_audio(y, sr)
        if not validation["valid"]:
            for issue in validation["issues"]:
                logger.warning("Audio issue: %s", issue)

        logger.info("Preprocessing complete: duration=%.2fs, sr=%d", len(y) / sr, sr)
        return y, sr
