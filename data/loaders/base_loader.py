"""Abstract base class for Voice Biomarker Platform v2 data loaders.

All dataset loaders (synthetic, mPower, DAIC-WOZ, etc.) should subclass
:class:`BaseLoader` and implement its abstract methods.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from data.loaders.utils import (
    FeatureCache,
    LabelEncoder,
    check_audio_directory,
    stratified_split,
    validate_audio_file,
)

logger = logging.getLogger(__name__)


class BaseLoader(ABC):
    """Abstract base class for all data loaders.

    Subclasses must implement:
    - :meth:`load_dataset`
    - :meth:`get_splits`
    - :meth:`get_metadata`

    The base class provides common utilities for audio loading, label
    encoding, train/val/test splitting with stratification, and
    caching.

    Parameters
    ----------
    data_dir:
        Root directory for this dataset (e.g. ``data/synthetic/``).
    target_sr:
        Target sample rate for loading audio (samples are resampled on load).
    max_duration:
        Maximum audio duration in seconds (shorter clips are zero-padded).
    min_duration:
        Minimum audio duration in seconds (shorter clips are discarded).
    cache_dir:
        Optional directory for caching extracted features.
    seed:
        Random seed for reproducible splits.
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        *,
        target_sr: int = 16000,
        max_duration: int = 30,
        min_duration: float = 0.5,
        cache_dir: Optional[Union[str, Path]] = None,
        seed: int = 42,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.seed = seed

        self._label_encoder = LabelEncoder()
        self._cache: Optional[FeatureCache] = None
        if cache_dir is not None:
            self._cache = FeatureCache(cache_dir)

        self._samples: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load and return the full dataset.

        Returns
        -------
        list of dict
            Each dict must contain at least ``"audio"`` (np.ndarray),
            ``"label"`` (int), and ``"label_name"`` (str) keys.
        """
        ...

    @abstractmethod
    def get_splits(
        self,
        *,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return stratified train/val/test splits.

        Returns
        -------
        dict
            ``{"train": [...], "val": [...], "test": [...]}`` where each
            value is a list of sample dicts.
        """
        ...

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset-level metadata (counts, class distribution, etc.)."""
        ...

    # ------------------------------------------------------------------
    # Concrete helper methods
    # ------------------------------------------------------------------

    @property
    def label_encoder(self) -> LabelEncoder:
        """Return the label encoder (call :meth:`load_dataset` first)."""
        return self._label_encoder

    @property
    def num_classes(self) -> int:
        return self._label_encoder.num_classes

    @property
    def class_names(self) -> List[str]:
        return self._label_encoder.classes

    def _ensure_loaded(self) -> List[Dict[str, Any]]:
        """Load dataset lazily if not already loaded."""
        if self._samples is None:
            logger.info("Loading dataset from %s ...", self.data_dir)
            self._samples = self.load_dataset()
            logger.info("Loaded %d samples.", len(self._samples))
        return self._samples

    def get_samples(self) -> List[Dict[str, Any]]:
        """Return all samples, loading the dataset lazily."""
        return self._ensure_loaded()

    def get_labels(self) -> np.ndarray:
        """Return a 1-D integer array of all labels."""
        samples = self._ensure_loaded()
        return np.array([s["label"] for s in samples], dtype=np.int64)

    def get_class_weights(self) -> Dict[str, float]:
        """Compute inverse-frequency class weights.

        Returns
        -------
        dict
            ``{class_name: weight}`` where weights sum to ``num_classes``.
        """
        labels = self.get_labels()
        counts = np.bincount(labels, minlength=self.num_classes)
        weights = len(labels) / (self.num_classes * counts.astype(np.float64))
        return {
            self._label_encoder.decode(i): float(w)
            for i, w in enumerate(weights)
        }

    # ------------------------------------------------------------------
    # Audio loading utilities
    # ------------------------------------------------------------------

    def load_audio(
        self,
        filepath: Union[str, Path],
        *,
        sr: Optional[int] = None,
        duration: Optional[float] = None,
    ) -> np.ndarray:
        """Load an audio file as a float32 NumPy array.

        Parameters
        ----------
        filepath:
            Path to the audio file.
        sr:
            Target sample rate. Falls back to ``self.target_sr``.
        duration:
            Target duration in seconds. Clips or zero-pads to this length.

        Returns
        -------
        np.ndarray
            1-D float32 audio array.
        """
        target_sr = sr or self.target_sr
        target_len = int(target_sr * (duration or self.max_duration))

        try:
            import librosa

            audio, _ = librosa.load(str(filepath), sr=target_sr, mono=True)
        except ImportError:
            # Fallback: try loading WAV with the stdlib
            audio = self._load_wav_fallback(filepath, target_sr)

        # Clip or zero-pad to target length
        if len(audio) > target_len:
            audio = audio[:target_len]
        elif len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")

        return audio.astype(np.float32)

    @staticmethod
    def _load_wav_fallback(
        filepath: Union[str, Path], target_sr: int
    ) -> np.ndarray:
        """Fallback WAV loader using the Python stdlib."""
        import struct
        import wave

        with wave.open(str(filepath), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)

        if sampwidth == 2:
            dtype = np.int16
        elif sampwidth == 4:
            dtype = np.int32
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth}")

        audio = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels)[:, 0]

        # Simple resampling if needed
        if sr != target_sr:
            ratio = target_sr / sr
            n_new = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n_new),
                np.arange(len(audio)),
                audio,
            )

        # Normalize to [-1, 1]
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak

        return audio.astype(np.float32)

    # ------------------------------------------------------------------
    # Splitting utilities
    # ------------------------------------------------------------------

    def _stratified_split(
        self,
        samples: List[Dict[str, Any]],
        *,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Split *samples* into train/val/test using stratification."""
        labels = [s["label"] for s in samples]
        split_indices = stratified_split(
            labels,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=self.seed,
        )
        return {
            split: [samples[i] for i in idx_list]
            for split, idx_list in split_indices.items()
        }

    # ------------------------------------------------------------------
    # Caching utilities
    # ------------------------------------------------------------------

    def _get_cached_features(self, key: str) -> Optional[np.ndarray]:
        if self._cache is None:
            return None
        return self._cache.get(key)

    def _put_cached_features(self, key: str, features: np.ndarray) -> None:
        if self._cache is not None:
            self._cache.put(key, features)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def validate_samples(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Validate all loaded samples.

        Returns
        -------
        (valid_samples, errors)
        """
        samples = self._ensure_loaded()
        valid: List[Dict[str, Any]] = []
        errors: List[str] = []

        for i, sample in enumerate(samples):
            audio = sample.get("audio")
            if audio is None:
                errors.append(f"Sample {i}: missing 'audio' key")
                continue
            if not isinstance(audio, np.ndarray):
                errors.append(f"Sample {i}: 'audio' is not a NumPy array")
                continue
            if audio.ndim != 1:
                errors.append(f"Sample {i}: audio must be 1-D, got {audio.ndim}-D")
                continue
            if "label" not in sample:
                errors.append(f"Sample {i}: missing 'label' key")
                continue
            valid.append(sample)

        return valid, errors

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"data_dir='{self.data_dir}', "
            f"target_sr={self.target_sr}, "
            f"seed={self.seed})"
        )
