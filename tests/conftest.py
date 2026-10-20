"""Shared test fixtures for Voice Biomarker Platform v2 tests."""

from __future__ import annotations

import json
import shutil
import tempfile
import wave
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    """Temporary output directory that is cleaned up after each test."""
    out = tmp_path / "output"
    out.mkdir()
    return out


@pytest.fixture
def tmp_models_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models" / "checkpoints"
    d.mkdir(parents=True)
    return d


# ---------------------------------------------------------------------------
# Audio fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_audio_16k() -> np.ndarray:
    """3-second sine wave at 440 Hz, 16 kHz sample rate."""
    sr = 16000
    t = np.linspace(0, 3.0, sr * 3, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def sample_audio_22k() -> np.ndarray:
    """1-second sine wave at 440 Hz, 22050 Hz sample rate."""
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


@pytest.fixture
def noisy_audio() -> np.ndarray:
    """1-second noisy audio at 16 kHz."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = np.random.default_rng(42).normal(0, 0.1, sr)
    return (signal + noise).astype(np.float32)


@pytest.fixture
def silence_audio() -> np.ndarray:
    """1-second near-silent audio."""
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def short_audio() -> np.ndarray:
    """0.1-second very short audio."""
    sr = 16000
    return np.sin(np.linspace(0, 2 * np.pi * 440 * 0.1, int(sr * 0.1))).astype(np.float32)


@pytest.fixture
def long_audio() -> np.ndarray:
    """35-second audio (exceeds typical max duration)."""
    sr = 16000
    t = np.linspace(0, 35.0, sr * 35, endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# WAV file fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_wav_path(tmp_path: Path, sample_audio_16k: np.ndarray) -> Path:
    """Create a valid WAV file and return its path."""
    filepath = tmp_path / "test_audio.wav"
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        audio_int16 = (sample_audio_16k * 32767).astype(np.int16)
        wf.writeframes(audio_int16.tobytes())
    return filepath


@pytest.fixture
def invalid_wav_path(tmp_path: Path) -> Path:
    """Create a corrupt WAV file."""
    filepath = tmp_path / "corrupt.wav"
    filepath.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt GARBAGE_DATA_HERE")
    return filepath


@pytest.fixture
def empty_wav_path(tmp_path: Path) -> Path:
    """Create an empty WAV file."""
    filepath = tmp_path / "empty.wav"
    filepath.write_bytes(b"")
    return filepath


@pytest.fixture
def mp3_wav_path(tmp_path: Path) -> Path:
    """Create a file with unsupported extension."""
    filepath = tmp_path / "test.xyz"
    filepath.write_bytes(b"not_audio_data")
    return filepath


# ---------------------------------------------------------------------------
# Feature fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_features() -> Dict[str, float]:
    """Sample feature dictionary mimicking FeatureExtractor output."""
    rng = np.random.default_rng(42)
    features = {
        "jitter_local": 0.005,
        "jitter_rap": 0.003,
        "jitter_ppq5": 0.004,
        "shimmer_local": 0.03,
        "shimmer_apq3": 0.025,
        "shimmer_apq5": 0.028,
        "hnr_db": 15.0,
        "f0_mean": 150.0,
        "f0_std": 25.0,
        "f0_min": 100.0,
        "f0_max": 200.0,
        "voicing_fraction": 0.85,
        "n_pauses": 3,
        "pause_ratio": 0.1,
        "total_pause_duration": 0.5,
        "mean_pause_duration": 0.167,
        "max_pause_duration": 0.25,
        "min_pause_duration": 0.1,
        "std_pause_duration": 0.05,
        "speech_rate": 4.0,
        "articulation_rate": 4.5,
        "n_onset_events": 20,
        "total_duration": 3.0,
        "speech_duration": 2.5,
        "spectral_centroid_mean": 1500.0,
        "spectral_bandwidth_mean": 1200.0,
        "spectral_rolloff_mean": 3000.0,
        "spectral_flatness_mean": 0.02,
        "zcr_mean": 0.1,
        "rms_mean": 0.05,
        "rms_std": 0.02,
        "rms_max": 0.15,
        "pd_f0_mean": 150.0,
        "pd_f0_std": 25.0,
        "pd_speech_rate": 4.0,
        "pd_pause_ratio": 0.1,
        "dep_f0_std": 25.0,
        "dep_f0_range": 100.0,
        "dep_pause_ratio": 0.1,
        "dep_speech_rate": 4.0,
        "dep_spectral_flatness": 0.02,
        "dep_rms_std": 0.03,
        "dep_voicing_fraction": 0.85,
        "dep_mean_pause_duration": 0.167,
    }
    for i in range(13):
        for suffix in ("mean", "std", "min", "max"):
            features[f"mfcc_{i}_{suffix}"] = float(rng.uniform(-10, 10))
    return features


@pytest.fixture
def anomalous_features() -> Dict[str, float]:
    """Features that should trigger anomaly detection (Parkinson's + Depression)."""
    features = {
        "jitter_local": 0.03,
        "shimmer_local": 0.12,
        "hnr_db": 4.0,
        "pd_f0_std": 4.0,
        "pd_speech_rate": 1.5,
        "pd_pause_ratio": 0.5,
        "dep_f0_std": 5.0,
        "dep_f0_range": 10.0,
        "dep_spectral_flatness": 0.3,
        "dep_pause_ratio": 0.5,
        "dep_mean_pause_duration": 1.5,
        "dep_speech_rate": 1.0,
        "dep_rms_std": 0.003,
        "dep_voicing_fraction": 0.5,
        "dep_f0_mean": 120.0,
        "f0_mean": 120.0,
        "f0_std": 15.0,
        "f0_min": 80.0,
        "f0_max": 160.0,
        "voicing_fraction": 0.6,
        "n_pauses": 8,
        "pause_ratio": 0.4,
        "total_pause_duration": 2.0,
        "mean_pause_duration": 0.5,
        "max_pause_duration": 1.2,
        "min_pause_duration": 0.2,
        "std_pause_duration": 0.3,
        "speech_rate": 1.5,
        "articulation_rate": 2.0,
        "n_onset_events": 10,
        "total_duration": 4.0,
        "speech_duration": 2.0,
    }
    for i in range(13):
        for suffix in ("mean", "std", "min", "max"):
            features[f"mfcc_{i}_{suffix}"] = float(np.random.default_rng(i).uniform(-10, 10))
    return features


@pytest.fixture
def sample_features_array() -> np.ndarray:
    """Feature matrix (10 samples, 50 features)."""
    rng = np.random.default_rng(42)
    return rng.random((10, 50)).astype(np.float64)


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Label array for 10 samples (binary classification)."""
    return np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64)


@pytest.fixture
def sample_features_2d() -> np.ndarray:
    """Feature matrix (20 samples, 50 features) for more robust testing."""
    rng = np.random.default_rng(42)
    return rng.random((20, 50)).astype(np.float64)


@pytest.fixture
def sample_labels_2d() -> np.ndarray:
    """Label array for 20 samples."""
    return np.array([0]*10 + [1]*10, dtype=np.int64)


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_classical_classifier():
    """A mock ClassicalClassifier with pre-set predict/predict_proba."""
    from models.classical_models import ClassicalClassifier

    classifier = MagicMock(spec=ClassicalClassifier)
    classifier.is_fitted = True
    classifier._model_type = "xgboost"
    classifier._n_classes = 2
    classifier._classes = np.array([0, 1])

    classifier.predict.return_value = np.array([0, 1, 0, 1, 1])
    classifier.predict_proba.return_value = np.array([
        [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.2, 0.8],
    ])
    classifier.evaluate.return_value = {
        "accuracy": 0.8,
        "f1_macro": 0.75,
        "f1_weighted": 0.78,
        "precision_macro": 0.77,
        "recall_macro": 0.75,
        "roc_auc": 0.85,
    }
    classifier.feature_importances_ = np.random.default_rng(42).random(50)
    classifier.get_feature_importance.return_value = classifier.feature_importances_

    return classifier


# ---------------------------------------------------------------------------
# API fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    """Mock settings for testing without real config."""
    settings = MagicMock()
    settings.audio.target_sr = 16000
    settings.audio.max_duration = 30
    settings.audio.min_duration = 2
    settings.audio.quality_threshold = 0.3
    settings.model.model_type = "xgboost"
    settings.model.checkpoint_dir = "models/checkpoints"
    settings.model.use_deep_model = False
    settings.training.epochs = 5
    settings.training.learning_rate = 0.001
    settings.training.batch_size = 16
    settings.training.early_stopping_patience = 3
    settings.explainability.use_shap = False
    settings.explainability.use_lime = False
    settings.explainability.num_features = 10
    settings.data.synthetic_samples = 50
    settings.data.data_dir = "data"
    settings.base_dir = str(Path(__file__).resolve().parent.parent)
    return settings


@pytest.fixture
def api_client():
    """Create a FastAPI test client."""
    from fastapi.testclient import TestClient
    from api.main import app
    from api import database as db
    db.init_db()
    # Create a test user matching the API key user_id so FK constraints pass
    try:
        with db.get_db() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO users (id, username, password_hash, role, is_active, created_at) "
                "VALUES (?, ?, ?, ?, 1, ?)",
                ("test-user", "test-user", "hashed", "clinician", "2025-01-01T00:00:00"),
            )
    except Exception:
        pass
    return TestClient(app)


@pytest.fixture
def mock_identity() -> Dict[str, Any]:
    """Mock authenticated user identity."""
    return {
        "user_id": "test-user-001",
        "role": "clinician",
        "method": "jwt",
    }


@pytest.fixture
def admin_identity() -> Dict[str, Any]:
    """Mock admin user identity."""
    return {
        "user_id": "admin-001",
        "role": "admin",
        "method": "jwt",
    }


# ---------------------------------------------------------------------------
# Sample audio data
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_synthetic_samples() -> List[Dict[str, Any]]:
    """List of sample dicts mimicking SyntheticLoader output."""
    rng = np.random.default_rng(42)
    samples = []
    for i in range(20):
        audio = rng.random(16000 * 2).astype(np.float32) * 0.5
        label = i % 2
        samples.append({
            "id": i,
            "audio": audio,
            "label": label,
            "label_name": "healthy" if label == 0 else "parkinson",
            "duration_sec": 2.0,
            "source": "synthetic",
        })
    return samples


@pytest.fixture
def sample_diagnosis_result() -> Dict[str, Any]:
    """Sample diagnosis agent output."""
    return {
        "parkinsons": {
            "prediction": "negative",
            "predicted_class": 0,
            "risk_score": 0.15,
            "risk_level": "low",
            "confidence": 0.85,
            "method": "rule_based",
            "model_type": None,
        },
        "depression": {
            "prediction": "negative",
            "predicted_class": 0,
            "risk_score": 0.1,
            "risk_level": "low",
            "confidence": 0.9,
            "method": "rule_based",
            "model_type": None,
        },
        "escalation_needed": False,
        "recommendations": [
            "No significant voice biomarker anomalies detected."
        ],
    }


@pytest.fixture
def sample_biomarker_result() -> Dict[str, Any]:
    """Sample biomarker agent output."""
    return {
        "parkinsons": {
            "risk_score": 0.15,
            "risk_level": "low",
            "contributing_features": [],
            "num_anomalies": 0,
        },
        "depression": {
            "risk_score": 0.1,
            "risk_level": "low",
            "contributing_features": [],
            "num_anomalies": 0,
        },
        "anomalies": [],
        "anomaly_details": [],
        "biomarker_scores": {
            "parkinsons_risk": 0.15,
            "depression_risk": 0.1,
        },
    }


@pytest.fixture
def sample_preprocessing_result(
    sample_audio_16k: np.ndarray, sample_features: Dict[str, float],
) -> Dict[str, Any]:
    """Sample preprocessing agent output."""
    return {
        "audio": sample_audio_16k,
        "sample_rate": 16000,
        "features": sample_features,
        "quality_metrics": {
            "snr_db": 30.0,
            "rms": 0.1,
            "peak": 0.5,
            "quality_score": 0.8,
        },
        "metadata": {
            "filepath": "/tmp/test.wav",
            "filename": "test.wav",
            "format": ".wav",
            "duration": 1.0,
            "sample_rate": 16000,
            "num_samples": 16000,
            "quality_score": 0.8,
            "snr_db": 30.0,
            "peak": 0.5,
            "rms": 0.1,
        },
        "validation_report": "Validation Report\nOverall Status: PASS",
    }


@pytest.fixture
def sample_explainability_result() -> Dict[str, Any]:
    """Sample explainability agent output."""
    return {
        "shap_explanation": None,
        "lime_explanation": None,
        "feature_importance": [
            {"feature": "jitter_local", "importance": 0.15, "relative_pct": 25.0},
            {"feature": "hnr_db", "importance": 0.12, "relative_pct": 20.0},
        ],
        "top_features": [
            {"feature": "jitter_local", "relevance_score": 0.9, "rank": 1},
            {"feature": "hnr_db", "relevance_score": 0.7, "rank": 2},
        ],
        "plain_language_explanation": "Analysis detected low risk for both conditions.",
    }
