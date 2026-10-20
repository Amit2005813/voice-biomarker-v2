"""Data loaders for the Voice Biomarker Platform v2.

Provides a unified interface for loading synthetic, mPower, and DAIC-WOZ
datasets through the :class:`BaseLoader` abstraction.

Usage
-----
::

    from data.loaders import SyntheticLoader, MPowerLoader, DAICWOZLoader

    loader = SyntheticLoader("data/synthetic", num_samples=1500)
    loader.load_dataset()
    splits = loader.get_splits()

    loader = MPowerLoader("data/mpower", task="voice")
    loader.load_dataset()

    loader = DAICWOZLoader("data/daicwoz", label_mode="binary")
    loader.load_dataset()
"""

from data.loaders.base_loader import BaseLoader
from data.loaders.daicwoz_loader import DAICWOZLoader, phq8_binary, phq8_label
from data.loaders.mpower_loader import MPowerLoader
from data.loaders.synthetic_loader import SyntheticLoader
from data.loaders.utils import (
    FeatureCache,
    LabelEncoder,
    add_noise,
    pitch_shift,
    random_augment,
    save_json,
    load_json,
    stratified_split,
    time_stretch,
    validate_audio_file,
    volume_normalize,
    AudioValidationError,
)

__all__ = [
    # Core loaders
    "BaseLoader",
    "SyntheticLoader",
    "MPowerLoader",
    "DAICWOZLoader",
    # Utility classes
    "LabelEncoder",
    "FeatureCache",
    # Augmentation
    "add_noise",
    "pitch_shift",
    "random_augment",
    "time_stretch",
    "volume_normalize",
    # Validation
    "validate_audio_file",
    "AudioValidationError",
    # Label helpers
    "phq8_binary",
    "phq8_label",
    # I/O
    "save_json",
    "load_json",
    # Splitting
    "stratified_split",
]
