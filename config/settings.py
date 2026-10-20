"""Central configuration for Voice Biomarker Platform v2.

Uses pydantic BaseSettings for environment variable support and
dataclasses for structured configuration groups.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


# ---------------------------------------------------------------------------
# Dataclass-based configuration groups
# ---------------------------------------------------------------------------

@dataclass
class AudioConfig:
    """Audio processing parameters."""

    target_sr: int = 16000
    max_duration: int = 30
    min_duration: int = 2
    quality_threshold: float = 0.3


@dataclass
class ModelConfig:
    """Machine learning model settings."""

    model_type: str = "xgboost"
    num_features: int = 50
    use_deep_model: bool = False
    checkpoint_dir: str = "models/checkpoints"


@dataclass
class TrainingConfig:
    """Model training hyper-parameters."""

    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    validation_split: float = 0.2
    early_stopping_patience: int = 10


@dataclass
class APIConfig:
    """FastAPI server settings."""

    host: str = "0.0.0.0"
    port: int = 8000
    max_upload_size: int = 50 * 1024 * 1024  # 50 MB in bytes
    allowed_extensions: List[str] = field(
        default_factory=lambda: ["wav", "mp3", "ogg", "flac", "m4a"]
    )


@dataclass
class ExplainabilityConfig:
    """Model explainability settings."""

    use_shap: bool = True
    use_lime: bool = True
    num_features: int = 10


@dataclass
class DataConfig:
    """Data generation and storage settings."""

    synthetic_samples: int = 1000
    data_dir: str = "data"


# ---------------------------------------------------------------------------
# Pydantic BaseSettings – root configuration with env-var support
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Top-level settings object.

    Every field can be overridden via an environment variable by using the
    field name prefixed with ``VB2_`` (e.g. ``VB2_DEBUG=true``).  Nested
    dataclass groups are instantiated from their defaults and can be
    replaced wholesale if needed.
    """

    # Environment
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(default="development", description="Runtime environment name")
    log_level: str = Field(default="INFO", description="Logging level")

    # Project paths
    base_dir: str = Field(
        default_factory=lambda: str(Path(__file__).resolve().parent.parent),
        description="Absolute path to the project root",
    )

    # Nested configuration groups
    audio: AudioConfig = Field(default_factory=AudioConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    explainability: ExplainabilityConfig = Field(default_factory=ExplainabilityConfig)
    data: DataConfig = Field(default_factory=DataConfig)

    # Optional secrets / external services
    secret_key: Optional[str] = Field(default=None, description="JWT secret key")
    redis_url: Optional[str] = Field(default=None, description="Redis connection URL")

    model_config = {
        "env_prefix": "VB2_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Return (and cache) the global :class:`Settings` instance.

    The instance is created on first call so that environment variables are
    read only once.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
