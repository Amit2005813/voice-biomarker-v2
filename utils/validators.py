"""Input validation utilities for the Voice Biomarker Platform."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_AUDIO_EXTENSIONS: set[str] = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
}

SUPPORTED_MODEL_TYPES: set[str] = {
    "random_forest",
    "xgboost",
    "svm",
    "neural_network",
    "logistic_regression",
}

MAX_AUDIO_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB
MIN_AUDIO_SIZE_BYTES = 100  # 100 B (avoid empty / trivial files)

# ---------------------------------------------------------------------------
# Validation error
# ---------------------------------------------------------------------------


class ValidationError(Exception):
    """Raised when an input fails validation."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


# ---------------------------------------------------------------------------
# Audio file validation
# ---------------------------------------------------------------------------


def validate_audio_file(
    filepath: str | Path,
    *,
    max_size: int = MAX_AUDIO_SIZE_BYTES,
    allowed_extensions: set[str] | None = None,
) -> None:
    """Perform comprehensive validation on an audio file.

    Checks extension, existence, readability, and size bounds.

    Args:
        filepath: Path to the audio file.
        max_size: Maximum allowed file size in bytes.
        allowed_extensions: Set of acceptable extensions (with leading dot).

    Raises:
        ValidationError: If any check fails.
    """
    errors: list[str] = []
    p = Path(filepath)

    if allowed_extensions is None:
        allowed_extensions = SUPPORTED_AUDIO_EXTENSIONS

    # Existence
    if not p.exists():
        errors.append(f"File does not exist: {p}")
        raise ValidationError(errors)

    if not p.is_file():
        errors.append(f"Path is not a file: {p}")
        raise ValidationError(errors)

    # Extension
    if p.suffix.lower() not in allowed_extensions:
        errors.append(
            f"Unsupported extension '{p.suffix}'. "
            f"Allowed: {', '.join(sorted(allowed_extensions))}"
        )

    # Size
    size = p.stat().st_size
    if size < MIN_AUDIO_SIZE_BYTES:
        errors.append(f"File is too small ({size} bytes). Likely empty or corrupt.")
    elif size > max_size:
        errors.append(
            f"File exceeds maximum size ({size:,} bytes > {max_size:,} bytes)."
        )

    # Readability
    try:
        with open(p, "rb") as fh:
            fh.read(16)
    except PermissionError:
        errors.append(f"No read permission: {p}")
    except OSError as exc:
        errors.append(f"Cannot read file: {exc}")

    if errors:
        raise ValidationError(errors)


# ---------------------------------------------------------------------------
# Model configuration validation
# ---------------------------------------------------------------------------


def validate_model_config(config: dict[str, Any]) -> None:
    """Validate a model configuration dictionary.

    Expected keys:
        - ``model_type`` (str, required)
        - ``params`` (dict, optional)

    Raises:
        ValidationError: If the configuration is invalid.
    """
    errors: list[str] = []

    if not isinstance(config, dict):
        raise ValidationError(["Configuration must be a dictionary."])

    model_type = config.get("model_type")
    if not model_type:
        errors.append("'model_type' is required.")
    elif model_type not in SUPPORTED_MODEL_TYPES:
        errors.append(
            f"Unknown model_type '{model_type}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_MODEL_TYPES))}"
        )

    params = config.get("params", {})
    if not isinstance(params, dict):
        errors.append("'params' must be a dictionary.")

    if errors:
        raise ValidationError(errors)


# ---------------------------------------------------------------------------
# Training data validation
# ---------------------------------------------------------------------------


def validate_training_data(
    data: Any,
    *,
    min_samples: int = 10,
    require_labels: bool = True,
    label_column: str = "label",
) -> None:
    """Validate training data before model training.

    Accepts a ``pandas.DataFrame`` or any object exposing ``columns`` and
    ``__len__`` (duck-typing for DataFrames).

    Args:
        data: Training data (DataFrame-like).
        min_samples: Minimum number of rows required.
        require_labels: Whether a label column must be present.
        label_column: Name of the label column.

    Raises:
        ValidationError: If the data fails any check.
    """
    errors: list[str] = []

    # Basic shape check
    if not hasattr(data, "columns") or not hasattr(data, "__len__"):
        raise ValidationError(
            ["Training data must be a DataFrame-like object with 'columns' and '__len__'."]
        )

    n_rows = len(data)  # type: ignore[arg-type]
    n_cols = len(data.columns)  # type: ignore[arg-type]

    if n_rows < min_samples:
        errors.append(
            f"Insufficient samples: {n_rows} < {min_samples}."
        )

    if n_cols < 2:
        errors.append(
            f"Expected at least 2 columns (features + label), got {n_cols}."
        )

    if require_labels:
        cols_lower = {str(c).lower() for c in data.columns}  # type: ignore[arg-type]
        if label_column.lower() not in cols_lower:
            errors.append(
                f"Label column '{label_column}' not found. "
                f"Available: {', '.join(str(c) for c in data.columns)}"  # type: ignore[arg-type]
            )

    # Check for all-null columns
    if hasattr(data, "isnull"):
        null_counts = data.isnull().sum()  # type: ignore[union-attr]
        fully_null = null_counts[null_counts == len(data)].index.tolist()  # type: ignore[arg-type]
        if fully_null:
            errors.append(
                f"Fully-null columns detected: {', '.join(str(c) for c in fully_null)}"
            )

    if errors:
        raise ValidationError(errors)
