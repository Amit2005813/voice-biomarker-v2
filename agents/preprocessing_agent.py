"""Audio preprocessing agent for Voice Biomarker Platform v2."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from agents.base_agent import BaseAgent
from config.settings import get_settings
from preprocessing.audio_processor import AudioPreprocessor
from preprocessing.feature_extractor import FeatureExtractor
from preprocessing.validators import AudioValidator, Severity

logger = logging.getLogger(__name__)


class AudioQualityError(Exception):
    """Raised when audio quality is below the acceptable threshold."""


class PreprocessingAgent(BaseAgent):
    """Load, validate, preprocess audio and extract features.

    Pipeline steps:

    1. Validate audio file (format, duration, quality).
    2. Load and resample.
    3. Noise reduction.
    4. Trim silence.
    5. Normalize amplitude.
    6. Extract acoustic features.

    Parameters
    ----------
    quality_threshold : float, optional
        Minimum quality score (0-1) to accept. Defaults to config value.
    """

    def __init__(self, quality_threshold: Optional[float] = None) -> None:
        super().__init__(name="preprocessing")
        settings = get_settings()
        self._target_sr = settings.audio.target_sr
        self._quality_threshold = (
            quality_threshold if quality_threshold is not None
            else settings.audio.quality_threshold
        )

        self.preprocessor = AudioPreprocessor(target_sr=self._target_sr)
        self.extractor = FeatureExtractor()
        self.validator = AudioValidator(
            target_sr=self._target_sr,
            min_duration=settings.audio.min_duration,
            max_duration=settings.audio.max_duration,
            min_quality_score=self._quality_threshold,
        )

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _execute(self, filepath: Union[str, Path], **kwargs: Any) -> Dict[str, Any]:
        """Run the full preprocessing pipeline.

        Parameters
        ----------
        filepath : str or Path
            Path to the audio file.

        Returns
        -------
        dict
            Keys: ``audio``, ``sample_rate``, ``features``,
            ``quality_metrics``, ``metadata``, ``validation_report``.

        Raises
        ------
        AudioQualityError
            If the audio quality score is below the configured threshold.
        FileNotFoundError
            If the file does not exist.
        ValueError
            If format or duration is invalid.
        """
        filepath = Path(filepath)

        # 1. Validation
        self.logger.info("Validating %s", filepath.name)
        validation = self.validator.get_validation_report(filepath)
        if not validation.passed:
            issues = "; ".join(r.message for r in validation.failures)
            raise AudioQualityError(
                f"Audio validation failed: {issues}"
            )

        # 2. Load + preprocess
        self.logger.info("Preprocessing %s", filepath.name)
        audio, sr = self.preprocessor.preprocess(
            filepath, noise_reduce=True, normalize=True,
        )

        # 3. Post-load quality check
        quality = self.preprocessor.get_quality_score(audio, sr)
        if quality["quality_score"] < self._quality_threshold:
            raise AudioQualityError(
                f"Quality score {quality['quality_score']:.4f} is below "
                f"threshold {self._quality_threshold:.4f}. "
                f"SNR={quality['snr_db']:.1f}dB, peak={quality['peak']:.4f}. "
                f"Please record in a quieter environment or speak closer to the microphone."
            )

        # 4. Feature extraction
        self.logger.info("Extracting features from %s", filepath.name)
        features = self.extractor.extract_all(audio, sr)

        # 5. Build metadata
        metadata = self._build_metadata(filepath, sr, audio, quality)

        self.logger.info(
            "Preprocessing complete: %d features extracted, quality=%.4f",
            len(features), quality["quality_score"],
        )

        return {
            "audio": audio,
            "sample_rate": sr,
            "features": features,
            "quality_metrics": quality,
            "metadata": metadata,
            "validation_report": validation.summary(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata(
        filepath: Path,
        sr: int,
        audio: np.ndarray,
        quality: Dict[str, float],
    ) -> Dict[str, Any]:
        """Construct metadata dictionary."""
        return {
            "filepath": str(filepath),
            "filename": filepath.name,
            "format": filepath.suffix.lower(),
            "duration": round(len(audio) / sr, 4),
            "sample_rate": sr,
            "num_samples": len(audio),
            "quality_score": quality["quality_score"],
            "snr_db": quality["snr_db"],
            "peak": quality["peak"],
            "rms": quality["rms"],
        }

    def get_feature_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert flat feature dict to a numeric vector for ML models.

        Parameters
        ----------
        features : dict
            Output of ``FeatureExtractor.extract_all``.

        Returns
        -------
        np.ndarray of shape ``(1, n_features)``
        """
        scalar_keys = sorted(k for k, v in features.items() if isinstance(v, (int, float)))
        return np.array([[float(features[k]) for k in scalar_keys]])

    def get_feature_names(self, features: Dict[str, Any]) -> List[str]:
        """Return ordered feature names matching :meth:`get_feature_vector`."""
        return sorted(k for k, v in features.items() if isinstance(v, (int, float)))
