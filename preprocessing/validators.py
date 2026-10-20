"""Audio validation utilities for Voice Biomarker Platform v2."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import librosa
except ImportError:
    raise ImportError("librosa is required: pip install librosa")

SUPPORTED_FORMATS: Tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma")


class Severity(str, Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class ValidationResult:
    check_name: str
    severity: Severity
    message: str
    value: Any = None


@dataclass
class ValidationReport:
    filepath: str
    results: List[ValidationResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.severity != Severity.FAIL for r in self.results)

    @property
    def warnings(self) -> List[ValidationResult]:
        return [r for r in self.results if r.severity == Severity.WARNING]

    @property
    def failures(self) -> List[ValidationResult]:
        return [r for r in self.results if r.severity == Severity.FAIL]

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"Validation Report for: {self.filepath}",
            f"Overall Status: {status}",
            f"  Checks run: {len(self.results)}",
            f"  Passed: {len(self.results) - len(self.warnings) - len(self.failures)}",
            f"  Warnings: {len(self.warnings)}",
            f"  Failures: {len(self.failures)}",
            "",
        ]
        for r in self.results:
            tag = r.severity.value.upper()
            lines.append(f"  [{tag}] {r.check_name}: {r.message}")
        return "\n".join(lines)


class AudioValidator:
    """Validate audio files for downstream processing."""

    def __init__(
        self,
        min_duration: float = 0.5,
        max_duration: float = 600.0,
        target_sr: int = 22050,
        min_snr_db: float = 5.0,
        min_quality_score: float = 0.3,
    ) -> None:
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.target_sr = target_sr
        self.min_snr_db = min_snr_db
        self.min_quality_score = min_quality_score

    def validate_format(self, filepath: Union[str, Path]) -> ValidationResult:
        """Check whether the file extension is supported.

        Args:
            filepath: Path to the audio file.

        Returns:
            ValidationResult indicating pass or fail.
        """
        path = Path(filepath)
        ext = path.suffix.lower()
        exists = path.exists()

        if not exists:
            return ValidationResult(
                check_name="format",
                severity=Severity.FAIL,
                message=f"File does not exist: {filepath}",
            )

        if ext not in SUPPORTED_FORMATS:
            return ValidationResult(
                check_name="format",
                severity=Severity.FAIL,
                message=f"Unsupported format '{ext}'. Supported: {', '.join(SUPPORTED_FORMATS)}",
            )

        return ValidationResult(
            check_name="format",
            severity=Severity.PASS,
            message=f"Valid format: {ext}",
            value=ext,
        )

    def validate_duration(
        self,
        filepath: Union[str, Path],
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
    ) -> ValidationResult:
        """Check that audio duration falls within acceptable bounds.

        Args:
            filepath: Path to the audio file.
            min_duration: Minimum duration in seconds (overrides instance default).
            max_duration: Maximum duration in seconds (overrides instance default).

        Returns:
            ValidationResult with duration information.
        """
        lo = min_duration if min_duration is not None else self.min_duration
        hi = max_duration if max_duration is not None else self.max_duration

        try:
            duration = librosa.get_duration(path=str(filepath))
        except Exception as exc:
            return ValidationResult(
                check_name="duration",
                severity=Severity.FAIL,
                message=f"Could not read duration: {exc}",
            )

        if duration < lo:
            return ValidationResult(
                check_name="duration",
                severity=Severity.FAIL,
                message=f"Duration {duration:.2f}s is below minimum {lo:.2f}s",
                value=duration,
            )

        if duration > hi:
            return ValidationResult(
                check_name="duration",
                severity=Severity.FAIL,
                message=f"Duration {duration:.2f}s exceeds maximum {hi:.2f}s",
                value=duration,
            )

        severity = Severity.WARNING if duration < lo * 2 else Severity.PASS
        return ValidationResult(
            check_name="duration",
            severity=severity,
            message=f"Duration: {duration:.2f}s",
            value=duration,
        )

    def validate_quality(
        self,
        filepath: Union[str, Path],
        min_snr_db: Optional[float] = None,
        min_quality_score: Optional[float] = None,
    ) -> ValidationResult:
        """Assess SNR-based quality of the audio file.

        Args:
            filepath: Path to the audio file.
            min_snr_db: Minimum SNR in dB (overrides instance default).
            min_quality_score: Minimum quality score 0-1 (overrides instance default).

        Returns:
            ValidationResult with quality metrics.
        """
        snr_lo = min_snr_db if min_snr_db is not None else self.min_snr_db
        q_lo = min_quality_score if min_quality_score is not None else self.min_quality_score

        try:
            y, _ = librosa.load(str(filepath), sr=self.target_sr, mono=True)
        except Exception as exc:
            return ValidationResult(
                check_name="quality",
                severity=Severity.FAIL,
                message=f"Could not load audio for quality check: {exc}",
            )

        if len(y) == 0:
            return ValidationResult(
                check_name="quality",
                severity=Severity.FAIL,
                message="Audio array is empty",
            )

        signal_power = float(np.mean(y ** 2))
        noise_estimate = float(np.mean(np.sort(np.abs(y))[: max(1, len(y) // 10)] ** 2))

        if noise_estimate < 1e-12:
            snr_db = 60.0
        else:
            snr_db = 10.0 * np.log10(max(signal_power, 1e-12) / noise_estimate)

        rms = float(np.sqrt(signal_power))
        peak = float(np.max(np.abs(y)))
        quality_score = min(1.0, max(0.0, (snr_db + 10) / 50.0))

        issues: List[str] = []
        severity = Severity.PASS

        if snr_db < snr_lo:
            issues.append(f"SNR {snr_db:.1f}dB < {snr_lo:.1f}dB")
            severity = Severity.FAIL

        if quality_score < q_lo:
            issues.append(f"Quality score {quality_score:.2f} < {q_lo:.2f}")
            severity = Severity.FAIL

        if peak < 1e-4:
            issues.append("Signal amplitude is near zero")
            severity = Severity.FAIL

        message = "Quality OK" if not issues else "; ".join(issues)

        return ValidationResult(
            check_name="quality",
            severity=severity,
            message=message,
            value={
                "snr_db": round(snr_db, 2),
                "rms": round(rms, 6),
                "peak": round(peak, 6),
                "quality_score": round(quality_score, 4),
            },
        )

    def get_validation_report(self, filepath: Union[str, Path]) -> ValidationReport:
        """Run all validation checks and produce a comprehensive report.

        Args:
            filepath: Path to the audio file.

        Returns:
            ValidationReport with results of every check.
        """
        report = ValidationReport(filepath=str(filepath))

        report.results.append(self.validate_format(filepath))

        if report.passed:
            report.results.append(self.validate_duration(filepath))
            report.results.append(self.validate_quality(filepath))

        return report
