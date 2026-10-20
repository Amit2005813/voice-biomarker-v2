"""Tests for data loaders: synthetic loader, augmentation, train/val/test splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

from data.loaders.utils import (
    LabelEncoder,
    FeatureCache,
    add_noise,
    pitch_shift,
    random_augment,
    random_crop,
    save_json,
    load_json,
    stratified_split,
    time_stretch,
    validate_audio_file,
    volume_normalize,
    AudioValidationError,
    synthesize_formant_signal,
    generate_noise_burst,
)
from data.loaders.synthetic_loader import SyntheticLoader
from data.loaders.base_loader import BaseLoader


# ======================================================================
# LabelEncoder
# ======================================================================

class TestLabelEncoder:
    """Test label encoding and decoding."""

    def test_fit_and_encode(self) -> None:
        enc = LabelEncoder()
        enc.fit(["healthy", "parkinson", "depression"])
        assert enc.num_classes == 3
        assert enc.encode("depression") == 0
        assert enc.encode("healthy") == 1
        assert enc.encode("parkinson") == 2

    def test_encode_many(self) -> None:
        enc = LabelEncoder()
        enc.fit(["a", "b", "c"])
        result = enc.encode_many(["a", "b", "c", "a"])
        np.testing.assert_array_equal(result, [0, 1, 2, 0])

    def test_decode(self) -> None:
        enc = LabelEncoder()
        enc.fit(["x", "y"])
        assert enc.decode(0) == "x"
        assert enc.decode(1) == "y"

    def test_decode_many(self) -> None:
        enc = LabelEncoder()
        enc.fit(["a", "b"])
        result = enc.decode_many([0, 1, 0])
        assert result == ["a", "b", "a"]

    def test_unknown_label(self) -> None:
        enc = LabelEncoder()
        enc.fit(["a"])
        with pytest.raises(KeyError):
            enc.encode("unknown")

    def test_unknown_index(self) -> None:
        enc = LabelEncoder()
        enc.fit(["a"])
        with pytest.raises(KeyError):
            enc.decode(99)

    def test_roundtrip(self) -> None:
        enc = LabelEncoder()
        enc.fit(["cat", "dog", "bird"])
        for label in ["cat", "dog", "bird"]:
            idx = enc.encode(label)
            decoded = enc.decode(idx)
            assert decoded == label

    def test_classes_sorted(self) -> None:
        enc = LabelEncoder()
        enc.fit(["c", "a", "b"])
        assert enc.classes == ["a", "b", "c"]

    def test_to_dict_and_from_dict(self) -> None:
        enc = LabelEncoder()
        enc.fit(["alpha", "beta", "gamma"])
        d = enc.to_dict()
        enc2 = LabelEncoder.from_dict(d)
        assert enc2.encode("alpha") == enc.encode("alpha")
        assert enc2.decode(1) == enc.decode(1)

    def test_save_and_load(self, tmp_path: Path) -> None:
        enc = LabelEncoder()
        enc.fit(["x", "y"])
        path = tmp_path / "encoder.json"
        enc.save(path)
        enc2 = LabelEncoder.load(path)
        assert enc2.encode("x") == enc.encode("x")


# ======================================================================
# FeatureCache
# ======================================================================

class TestFeatureCache:
    """Test disk-based feature caching."""

    def test_put_and_get(self, tmp_path: Path) -> None:
        cache = FeatureCache(tmp_path / "cache")
        features = np.array([1.0, 2.0, 3.0])
        cache.put("key1", features)
        loaded = cache.get("key1")
        np.testing.assert_array_equal(loaded, features)

    def test_get_miss(self, tmp_path: Path) -> None:
        cache = FeatureCache(tmp_path / "cache")
        assert cache.get("nonexistent") is None

    def test_has(self, tmp_path: Path) -> None:
        cache = FeatureCache(tmp_path / "cache")
        cache.put("k", np.array([1.0]))
        assert cache.has("k") is True
        assert cache.has("nope") is False

    def test_clear(self, tmp_path: Path) -> None:
        cache = FeatureCache(tmp_path / "cache")
        cache.put("a", np.array([1.0]))
        cache.put("b", np.array([2.0]))
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None

    def test_len(self, tmp_path: Path) -> None:
        cache = FeatureCache(tmp_path / "cache")
        assert len(cache) == 0
        cache.put("x", np.array([1.0]))
        assert len(cache) == 1


# ======================================================================
# Augmentation functions
# ======================================================================

class TestTimeStretch:
    """Test time stretching."""

    def test_stretch(self) -> None:
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32)
        stretched = time_stretch(audio, rate=1.5)
        assert len(stretched) > len(audio)

    def test_stretch_same_rate(self) -> None:
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = time_stretch(audio, rate=1.0)
        np.testing.assert_array_equal(result, audio)


class TestPitchShift:
    """Test pitch shifting."""

    def test_shift(self) -> None:
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32)
        shifted = pitch_shift(audio, 16000, n_semitones=2.0)
        assert isinstance(shifted, np.ndarray)

    def test_shift_zero(self) -> None:
        audio = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = pitch_shift(audio, 16000, n_semitones=0.0)
        np.testing.assert_array_equal(result, audio)


class TestAddNoise:
    """Test noise addition."""

    def test_add_gaussian_noise(self) -> None:
        audio = np.ones(16000, dtype=np.float32) * 0.5
        noisy = add_noise(audio, noise_level=0.01, noise_type="gaussian")
        assert noisy.shape == audio.shape
        assert np.max(np.abs(noisy)) <= 1.0

    def test_add_uniform_noise(self) -> None:
        audio = np.ones(16000, dtype=np.float32) * 0.5
        noisy = add_noise(audio, noise_level=0.01, noise_type="uniform")
        assert noisy.shape == audio.shape

    def test_invalid_noise_type(self) -> None:
        audio = np.ones(100, dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown noise_type"):
            add_noise(audio, noise_type="invalid")

    def test_noisy_is_different(self) -> None:
        audio = np.ones(16000, dtype=np.float32) * 0.5
        noisy = add_noise(audio, noise_level=0.05)
        assert not np.array_equal(audio, noisy)


class TestVolumeNormalize:
    """Test volume normalization."""

    def test_normalize(self) -> None:
        audio = np.array([0.1, 0.5, 0.3], dtype=np.float32)
        normalized = volume_normalize(audio, target_peak=0.9)
        peak = np.max(np.abs(normalized))
        assert abs(peak - 0.9) < 0.01

    def test_normalize_silent(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        result = volume_normalize(audio)
        np.testing.assert_array_equal(result, audio)


class TestRandomCrop:
    """Test random cropping."""

    def test_crop(self) -> None:
        audio = np.arange(1000, dtype=np.float32)
        cropped = random_crop(audio, 500)
        assert cropped is not None
        assert len(cropped) == 500

    def test_crop_too_large(self) -> None:
        audio = np.arange(100, dtype=np.float32)
        result = random_crop(audio, 500)
        assert result is None


class TestRandomAugment:
    """Test random augmentation pipeline."""

    def test_augment(self) -> None:
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32)
        augmented = random_augment(audio, sample_rate=16000)
        assert isinstance(augmented, np.ndarray)
        assert augmented.ndim == 1

    def test_augment_with_crop(self) -> None:
        audio = np.sin(np.linspace(0, 2 * np.pi * 440, 16000)).astype(np.float32)
        augmented = random_augment(
            audio, sample_rate=16000, prob_crop=1.0, target_samples=8000,
        )
        assert len(augmented) <= 16000


# ======================================================================
# Stratified split
# ======================================================================

class TestStratifiedSplit:
    """Test stratified train/val/test splitting."""

    def test_split_basic(self) -> None:
        labels = [0] * 100 + [1] * 100
        splits = stratified_split(labels, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 200

    def test_split_preserves_proportions(self) -> None:
        labels = [0] * 100 + [1] * 100
        splits = stratified_split(labels, seed=42)
        train = splits["train"]
        labels_train = [labels[i] for i in train]
        ratio_1 = labels_train.count(1) / len(labels_train)
        assert 0.3 < ratio_1 < 0.7

    def test_split_ratios_sum_to_one(self) -> None:
        labels = list(range(50)) * 2
        splits = stratified_split(labels)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == len(labels)

    def test_split_invalid_ratios(self) -> None:
        with pytest.raises(AssertionError):
            stratified_split([0, 1, 0, 1], train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)

    def test_split_reproducible(self) -> None:
        labels = list(range(100))
        s1 = stratified_split(labels, seed=42)
        s2 = stratified_split(labels, seed=42)
        assert s1["train"] == s2["train"]


# ======================================================================
# SyntheticLoader
# ======================================================================

class TestSyntheticLoader:
    """Test synthetic data generation."""

    def test_load_dataset(self, tmp_path: Path) -> None:
        loader = SyntheticLoader(
            data_dir=tmp_path / "syn",
            num_samples=20,
            seed=42,
            balanced=True,
        )
        samples = loader.load_dataset()
        assert len(samples) > 0
        assert "audio" in samples[0]
        assert "label" in samples[0]
        assert "label_name" in samples[0]

    def test_get_splits(self, tmp_path: Path) -> None:
        loader = SyntheticLoader(
            data_dir=tmp_path / "syn",
            num_samples=20,
            seed=42,
        )
        loader.load_dataset()
        splits = loader.get_splits()
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 20

    def test_balanced(self, tmp_path: Path) -> None:
        loader = SyntheticLoader(
            data_dir=tmp_path / "syn",
            num_samples=20,
            balanced=True,
            seed=42,
        )
        samples = loader.load_dataset()
        class_counts = {}
        for s in samples:
            name = s["label_name"]
            class_counts[name] = class_counts.get(name, 0) + 1
        counts = list(class_counts.values())
        assert max(counts) - min(counts) <= 1

    def test_unbalanced(self, tmp_path: Path) -> None:
        loader = SyntheticLoader(
            data_dir=tmp_path / "syn",
            num_samples=20,
            balanced=False,
            seed=42,
        )
        samples = loader.load_dataset()
        class_counts = {}
        for s in samples:
            name = s["label_name"]
            class_counts[name] = class_counts.get(name, 0) + 1
        assert len(class_counts) > 0

    def test_metadata(self, tmp_path: Path) -> None:
        loader = SyntheticLoader(
            data_dir=tmp_path / "syn",
            num_samples=20,
            seed=42,
        )
        loader.load_dataset()
        meta = loader.get_metadata()
        assert "total_samples" in meta
        assert "class_counts" in meta
        assert "class_names" in meta

    def test_persistence(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "syn"
        loader1 = SyntheticLoader(data_dir=data_dir, num_samples=10, seed=42)
        samples1 = loader1.load_dataset()
        loader2 = SyntheticLoader(data_dir=data_dir, num_samples=10, seed=42)
        samples2 = loader2.load_dataset()
        assert len(samples1) == len(samples2)

    def test_labels_path_exists(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "syn"
        loader = SyntheticLoader(data_dir=data_dir, num_samples=10, seed=42)
        loader.load_dataset()
        assert (data_dir / "labels.json").exists()

    def test_audio_files_on_disk(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "syn"
        loader = SyntheticLoader(data_dir=data_dir, num_samples=10, seed=42)
        loader.load_dataset()
        npy_files = list(data_dir.glob("*.npy"))
        assert len(npy_files) == 10


# ======================================================================
# Audio validation utilities
# ======================================================================

class TestValidateAudioFile:
    """Test audio file validation."""

    def test_validate_valid_wav(self, sample_wav_path: Path) -> None:
        meta = validate_audio_file(sample_wav_path, min_duration_sec=0.5, max_duration_sec=5.0)
        assert "duration_sec" in meta
        assert "sample_rate" in meta

    def test_validate_nonexistent(self) -> None:
        with pytest.raises(AudioValidationError, match="not found"):
            validate_audio_file("/nonexistent/file.wav")

    def test_validate_empty_file(self, empty_wav_path: Path) -> None:
        with pytest.raises(AudioValidationError, match="empty"):
            validate_audio_file(empty_wav_path)

    def test_validate_unsupported_format(self, mp3_wav_path: Path) -> None:
        with pytest.raises(AudioValidationError, match="Unsupported"):
            validate_audio_file(mp3_wav_path)

    def test_validate_too_short(self, sample_wav_path: Path) -> None:
        with pytest.raises(AudioValidationError, match="too short"):
            validate_audio_file(sample_wav_path, min_duration_sec=10.0)

    def test_validate_sample_rate_mismatch(self, sample_wav_path: Path) -> None:
        with pytest.raises(AudioValidationError, match="Sample rate mismatch"):
            validate_audio_file(sample_wav_path, expected_sr=44100)


# ======================================================================
# Audio synthesis
# ======================================================================

class TestAudioSynthesis:
    """Test audio synthesis helpers."""

    def test_synthesize_formant(self) -> None:
        rng = np.random.default_rng(42)
        audio = synthesize_formant_signal(
            duration_sec=1.0,
            sample_rate=16000,
            fundamental_freq=150.0,
            formant_freqs=[700, 1200, 2600],
            formant_bws=[80, 90, 120],
            rng=rng,
        )
        assert isinstance(audio, np.ndarray)
        assert len(audio) == 16000
        assert audio.dtype == np.float32
        peak = np.max(np.abs(audio))
        assert peak > 0

    def test_generate_noise_burst(self) -> None:
        rng = np.random.default_rng(42)
        burst = generate_noise_burst(0.1, 16000, rng=rng)
        assert isinstance(burst, np.ndarray)
        assert len(burst) == int(0.1 * 16000)

    def test_synthesize_zero_duration(self) -> None:
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError):
            synthesize_formant_signal(
                duration_sec=0.0,
                sample_rate=16000,
                fundamental_freq=150.0,
                formant_freqs=[700],
                formant_bws=[80],
                rng=rng,
            )


# ======================================================================
# JSON I/O
# ======================================================================

class TestJsonIO:
    """Test JSON save/load utilities."""

    def test_save_and_load(self, tmp_path: Path) -> None:
        data = {"key": "value", "nested": [1, 2, 3]}
        path = tmp_path / "test.json"
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_save_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "a" / "b" / "c.json"
        save_json({"x": 1}, path)
        assert path.exists()
