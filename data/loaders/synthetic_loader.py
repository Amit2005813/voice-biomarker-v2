"""Enhanced synthetic data generator for the Voice Biomarker Platform v2.

Generates realistic speech-like signals with formant structure, harmonic
content, and noise variations.  Produces balanced or unbalanced class
distributions with proper labels and saves everything to ``data/synthetic/``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from data.loaders.base_loader import BaseLoader
from data.loaders.utils import (
    add_noise,
    save_json,
    synthesize_formant_signal,
    generate_noise_burst,
    volume_normalize,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Predefined voice profiles for different conditions
# ---------------------------------------------------------------------------

_VoiceProfile = Dict[str, Any]

_PROFILES: Dict[str, _VoiceProfile] = {
    "healthy": {
        "f0_range": (110, 280),
        "formant_freqs": [[700, 1200, 2600], [550, 1050, 2400], [650, 1150, 2500]],
        "formant_bws": [[80, 90, 120], [70, 80, 110], [75, 85, 115]],
        "jitter_pct": 0.4,
        "shimmer_pct": 2.5,
        "duration_range": (2.0, 6.0),
        "noise_level": 0.001,
    },
    "parkinson": {
        "f0_range": (100, 220),
        "formant_freqs": [[600, 1000, 2300], [500, 900, 2100], [550, 950, 2200]],
        "formant_bws": [[100, 120, 160], [90, 110, 150], [95, 115, 155]],
        "jitter_pct": 1.8,
        "shimmer_pct": 8.0,
        "duration_range": (1.5, 4.0),
        "noise_level": 0.005,
    },
    "depression": {
        "f0_range": (90, 200),
        "formant_freqs": [[620, 1050, 2400], [580, 980, 2300], [600, 1020, 2350]],
        "formant_bws": [[90, 100, 140], [85, 95, 130], [88, 98, 135]],
        "jitter_pct": 1.2,
        "shimmer_pct": 5.5,
        "duration_range": (2.0, 5.0),
        "noise_level": 0.003,
    },
    "laryngitis": {
        "f0_range": (80, 180),
        "formant_freqs": [[450, 900, 2100], [400, 850, 2000], [425, 875, 2050]],
        "formant_bws": [[120, 140, 180], [110, 130, 170], [115, 135, 175]],
        "jitter_pct": 3.5,
        "shimmer_pct": 12.0,
        "duration_range": (1.5, 4.5),
        "noise_level": 0.008,
    },
}


class SyntheticLoader(BaseLoader):
    """Generate synthetic speech-like audio data with realistic voice profiles.

    Parameters
    ----------
    data_dir:
        Where to store / load the synthetic dataset.
    num_samples:
        Total number of samples to generate per class.
    target_sr:
        Sample rate for generated audio.
    max_duration:
        Maximum duration in seconds per sample.
    balanced:
        If *True*, each class gets the same number of samples.
    class_ratios:
        Optional override for class sample counts.  Ignored when
        ``balanced=True``.
    seed:
        Random seed.
    profiles:
        Custom voice profiles dict. Defaults to the built-in profiles.

    Examples
    --------
    >>> loader = SyntheticLoader("data/synthetic", num_samples=1500)
    >>> loader.load_dataset()
    >>> splits = loader.get_splits()
    >>> print(f"Train: {len(splits['train'])}, Val: {len(splits['val'])}")
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/synthetic",
        *,
        num_samples: int = 1000,
        target_sr: int = 16000,
        max_duration: int = 30,
        balanced: bool = True,
        class_ratios: Optional[Dict[str, int]] = None,
        seed: int = 42,
        profiles: Optional[Dict[str, _VoiceProfile]] = None,
    ) -> None:
        super().__init__(
            data_dir,
            target_sr=target_sr,
            max_duration=max_duration,
            seed=seed,
        )
        self.num_samples = num_samples
        self.balanced = balanced
        self.class_ratios = class_ratios
        self.profiles = profiles or _PROFILES

    # ------------------------------------------------------------------
    # BaseLoader interface
    # ------------------------------------------------------------------

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Generate (or reload) the full synthetic dataset.

        If ``data/synthetic/labels.json`` already exists the dataset is
        loaded from disk; otherwise it is generated from scratch.
        """
        labels_path = self.data_dir / "labels.json"

        if labels_path.exists():
            logger.info("Reloading existing synthetic data from %s", labels_path)
            return self._load_from_disk()

        logger.info("Generating %d synthetic samples ...", self.num_samples)
        samples = self._generate_all()
        self._save_to_disk(samples)
        return samples

    def get_splits(
        self,
        *,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return stratified train/val/test splits."""
        samples = self._ensure_loaded()
        return self._stratified_split(
            samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset metadata and class distribution."""
        samples = self._ensure_loaded()
        labels = [s["label"] for s in samples]
        class_counts: Dict[str, int] = {}
        for s in samples:
            name = s["label_name"]
            class_counts[name] = class_counts.get(name, 0) + 1

        return {
            "dataset": "synthetic_v2",
            "total_samples": len(samples),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "class_counts": class_counts,
            "target_sr": self.target_sr,
            "balanced": self.balanced,
            "profiles_used": list(self.profiles.keys()),
        }

    # ------------------------------------------------------------------
    # Generation internals
    # ------------------------------------------------------------------

    def _compute_class_counts(self) -> Dict[str, int]:
        """Determine how many samples to generate per class."""
        class_names = sorted(self.profiles.keys())

        if self.class_ratios is not None:
            return {k: self.class_ratios[k] for k in class_names if k in self.class_ratios}

        if self.balanced:
            per_class = self.num_samples // len(class_names)
            remainder = self.num_samples - per_class * len(class_names)
            counts = {name: per_class for name in class_names}
            # Distribute remainder
            for i in range(remainder):
                counts[class_names[i]] += 1
            return counts
        else:
            # Unbalanced: roughly 40% healthy, 20% each condition
            weights = np.array([0.4, 0.2, 0.2, 0.2])[: len(class_names)]
            weights = weights / weights.sum()
            counts = {}
            remaining = self.num_samples
            for i, name in enumerate(class_names):
                n = int(self.num_samples * weights[i])
                counts[name] = n
                remaining -= n
            # Assign remainder to first class
            counts[class_names[0]] += remaining
            return counts

    def _generate_all(self) -> List[Dict[str, Any]]:
        """Generate all samples."""
        rng = np.random.default_rng(self.seed)
        class_counts = self._compute_class_counts()
        self._label_encoder.fit(sorted(class_counts.keys()))

        samples: List[Dict[str, Any]] = []
        sample_id = 0

        for class_name, count in class_counts.items():
            profile = self.profiles[class_name]
            label_idx = self._label_encoder.encode(class_name)

            for j in range(count):
                audio = self._generate_single(profile, rng)
                samples.append(
                    {
                        "id": sample_id,
                        "audio": audio,
                        "label": label_idx,
                        "label_name": class_name,
                        "duration_sec": len(audio) / self.target_sr,
                        "source": "synthetic",
                        "params": {
                            "f0": profile["f0_range"],
                            "jitter": profile["jitter_pct"],
                            "shimmer": profile["shimmer_pct"],
                        },
                    }
                )
                sample_id += 1

        # Shuffle
        perm = rng.permutation(len(samples))
        return [samples[i] for i in perm]

    def _generate_single(
        self, profile: _VoiceProfile, rng: np.random.Generator
    ) -> np.ndarray:
        """Generate a single synthetic audio sample from a voice profile."""
        duration = float(
            rng.uniform(profile["duration_range"][0], profile["duration_range"][1])
        )
        f0 = float(rng.uniform(profile["f0_range"][0], profile["f0_range"][1]))

        # Pick a random formant configuration
        idx = int(rng.integers(0, len(profile["formant_freqs"])))
        formant_freqs = profile["formant_freqs"][idx]
        formant_bws = profile["formant_bws"][idx]

        # Synthesize vowel-like signal
        audio = synthesize_formant_signal(
            duration,
            self.target_sr,
            f0,
            formant_freqs,
            formant_bws,
            jitter_pct=profile["jitter_pct"]
            * float(rng.uniform(0.8, 1.2)),
            shimmer_pct=profile["shimmer_pct"]
            * float(rng.uniform(0.8, 1.2)),
            noise_level=profile["noise_level"],
            rng=rng,
        )

        # Mix in some "breathiness" (noise bursts)
        if rng.random() < 0.6:
            breath_dur = float(rng.uniform(0.05, 0.2))
            breath = generate_noise_burst(breath_dur, self.target_sr, rng=rng)
            # Insert at random position
            pos = int(rng.integers(0, max(1, len(audio) - len(breath))))
            end = min(pos + len(breath), len(audio))
            audio[pos:end] = 0.5 * audio[pos:end] + 0.5 * breath[: end - pos]

        # Add low-level broadband noise
        audio = add_noise(audio, noise_level=profile["noise_level"] * 0.5, rng=rng)

        # Normalize
        audio = volume_normalize(audio, target_peak=0.85)

        return audio

    # ------------------------------------------------------------------
    # Disk persistence
    # ------------------------------------------------------------------

    def _save_to_disk(self, samples: List[Dict[str, Any]]) -> None:
        """Save generated dataset to ``data/synthetic/``."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

        labels_meta: List[Dict[str, Any]] = []
        for i, s in enumerate(samples):
            audio_path = self.data_dir / f"sample_{i:05d}.npy"
            np.save(str(audio_path), s["audio"])
            labels_meta.append(
                {
                    "id": s["id"],
                    "audio_file": audio_path.name,
                    "label": s["label"],
                    "label_name": s["label_name"],
                    "duration_sec": s["duration_sec"],
                }
            )

        labels_path = self.data_dir / "labels.json"
        full_meta = {
            "dataset": "synthetic_v2",
            "total_samples": len(samples),
            "target_sr": self.target_sr,
            "label_encoder": self._label_encoder.to_dict(),
            "samples": labels_meta,
        }
        save_json(full_meta, labels_path)
        logger.info(
            "Saved %d synthetic samples to %s", len(samples), self.data_dir
        )

    def _load_from_disk(self) -> List[Dict[str, Any]]:
        """Load previously saved dataset from disk."""
        labels_path = self.data_dir / "labels.json"
        with open(labels_path, "r", encoding="utf-8") as fh:
            meta = json.load(fh)

        self._label_encoder = LabelEncoder.from_dict(meta["label_encoder"])
        self.target_sr = meta.get("target_sr", self.target_sr)

        samples: List[Dict[str, Any]] = []
        for entry in meta["samples"]:
            audio_path = self.data_dir / entry["audio_file"]
            if not audio_path.exists():
                logger.warning("Missing audio file: %s", audio_path)
                continue
            audio = np.load(str(audio_path))
            samples.append(
                {
                    "id": entry["id"],
                    "audio": audio,
                    "label": entry["label"],
                    "label_name": entry["label_name"],
                    "duration_sec": entry["duration_sec"],
                    "source": "synthetic",
                }
            )

        logger.info("Loaded %d samples from disk.", len(samples))
        return samples


# Import needed for _load_from_disk
from data.loaders.utils import LabelEncoder  # noqa: E402
