"""Biomarker analysis agent for Voice Biomarker Platform v2."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Thresholds for anomaly detection (evidence-based defaults)
# ------------------------------------------------------------------

PARKINSONS_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "jitter_local": {"high": 0.015, "very_high": 0.025},
    "shimmer_local": {"high": 0.05, "very_high": 0.10},
    "hnr_db": {"low": 10.0, "very_low": 5.0},
    "pd_speech_rate": {"low": 3.0, "very_low": 2.0},
    "pd_f0_std": {"low": 10.0, "very_low": 5.0},
    "pd_pause_ratio": {"high": 0.30, "very_high": 0.45},
}

DEPRESSION_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "dep_f0_std": {"low": 15.0, "very_low": 8.0},
    "dep_f0_range": {"low": 30.0, "very_low": 15.0},
    "dep_spectral_flatness": {"high": 0.10, "very_high": 0.25},
    "dep_pause_ratio": {"high": 0.25, "very_high": 0.40},
    "dep_mean_pause_duration": {"high": 0.5, "very_high": 1.0},
    "dep_speech_rate": {"low": 2.5, "very_low": 1.5},
    "dep_rms_std": {"low": 0.01, "very_low": 0.005},
}


class BiomarkerAgent(BaseAgent):
    """Analyze extracted acoustic features for disease biomarkers.

    Detects anomalies in voice quality measures and calculates
    composite biomarker scores for Parkinson's disease and depression.
    """

    def __init__(self) -> None:
        super().__init__(name="biomarker")

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _execute(self, features: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        """Analyze features and produce biomarker results.

        Parameters
        ----------
        features : dict
            Scalar feature dictionary from :class:`FeatureExtractor`.

        Returns
        -------
        dict
            Keys: ``parkinsons``, ``depression``, ``anomalies``,
            ``biomarker_scores``.
        """
        # 1. Detect anomalies across all biomarkers
        anomalies = self._detect_anomalies(features)

        # 2. Calculate composite scores
        parkinsons_score = self._score_parkinsons(features, anomalies)
        depression_score = self._score_depression(features, anomalies)

        # 3. Build per-anomaly explanations
        anomaly_details = self._build_anomaly_details(anomalies, features)

        results = {
            "parkinsons": parkinsons_score,
            "depression": depression_score,
            "anomalies": anomalies,
            "anomaly_details": anomaly_details,
            "biomarker_scores": {
                "parkinsons_risk": parkinsons_score["risk_score"],
                "depression_risk": depression_score["risk_score"],
            },
        }

        self.logger.info(
            "Biomarker analysis: PD risk=%.3f, Depression risk=%.3f, %d anomalies",
            parkinsons_score["risk_score"],
            depression_score["risk_score"],
            len(anomalies),
        )
        return results

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def _detect_anomalies(self, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan features against threshold tables and flag anomalies."""
        anomalies: List[Dict[str, Any]] = []

        # Parkinson's indicators
        for feat_key, thresholds in PARKINSONS_THRESHOLDS.items():
            value = features.get(feat_key)
            if value is None or not isinstance(value, (int, float)):
                continue

            for severity_name in ("very_high", "high", "very_low", "low"):
                bound = thresholds.get(severity_name)
                if bound is None:
                    continue

                triggered = False
                if "high" in severity_name and value > bound:
                    triggered = True
                if "low" in severity_name and value < bound:
                    triggered = True

                if triggered:
                    anomalies.append({
                        "feature": feat_key,
                        "value": round(float(value), 6),
                        "threshold": bound,
                        "severity": severity_name,
                        "condition": "parkinsons",
                        "direction": "high" if "high" in severity_name else "low",
                    })
                    break  # report worst severity only

        # Depression indicators
        for feat_key, thresholds in DEPRESSION_THRESHOLDS.items():
            value = features.get(feat_key)
            if value is None or not isinstance(value, (int, float)):
                continue

            for severity_name in ("very_high", "high", "very_low", "low"):
                bound = thresholds.get(severity_name)
                if bound is None:
                    continue

                triggered = False
                if "high" in severity_name and value > bound:
                    triggered = True
                if "low" in severity_name and value < bound:
                    triggered = True

                if triggered:
                    anomalies.append({
                        "feature": feat_key,
                        "value": round(float(value), 6),
                        "threshold": bound,
                        "severity": severity_name,
                        "condition": "depression",
                        "direction": "high" if "high" in severity_name else "low",
                    })
                    break

        return anomalies

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _score_parkinsons(
        features: Dict[str, Any],
        anomalies: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute a composite Parkinson's risk score (0-1).

        Weights are derived from clinical literature emphasizing jitter,
        shimmer, HNR, and speech rate as the strongest vocal biomarkers.
        """
        weights: Dict[str, float] = {
            "jitter_local": 0.20,
            "shimmer_local": 0.18,
            "hnr_db": 0.18,
            "pd_speech_rate": 0.14,
            "pd_f0_std": 0.12,
            "pd_pause_ratio": 0.10,
            "pd_spectral_flatness_mean": 0.08,
        }

        score = 0.0
        contributing: List[Dict[str, Any]] = []

        pd_anomalies = [a for a in anomalies if a["condition"] == "parkinsons"]
        anomaly_features = {a["feature"] for a in pd_anomalies}

        for feat_key, weight in weights.items():
            value = features.get(feat_key)
            if value is None or not isinstance(value, (int, float)):
                continue

            feat_anomaly = next(
                (a for a in pd_anomalies if a["feature"] == feat_key), None,
            )
            if feat_anomaly is not None:
                severity_mult = 1.0 if "very_" not in feat_anomaly["severity"] else 1.3
                score += weight * severity_mult
                contributing.append({
                    "feature": feat_key,
                    "weight": weight,
                    "contribution": round(weight * severity_mult, 4),
                    "value": feat_anomaly["value"],
                })

        risk_score = round(min(score, 1.0), 4)

        if risk_score >= 0.6:
            risk_level = "high"
        elif risk_score >= 0.3:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "contributing_features": sorted(contributing, key=lambda c: -c["contribution"]),
            "num_anomalies": len(pd_anomalies),
        }

    @staticmethod
    def _score_depression(
        features: Dict[str, Any],
        anomalies: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Compute a composite depression risk score (0-1).

        Focuses on prosodic flattening, pause patterns, and energy variability.
        """
        weights: Dict[str, float] = {
            "dep_f0_std": 0.16,
            "dep_f0_range": 0.14,
            "dep_spectral_flatness": 0.14,
            "dep_pause_ratio": 0.14,
            "dep_mean_pause_duration": 0.12,
            "dep_speech_rate": 0.12,
            "dep_rms_std": 0.10,
            "dep_voicing_fraction": 0.08,
        }

        score = 0.0
        contributing: List[Dict[str, Any]] = []

        dep_anomalies = [a for a in anomalies if a["condition"] == "depression"]

        for feat_key, weight in weights.items():
            value = features.get(feat_key)
            if value is None or not isinstance(value, (int, float)):
                continue

            feat_anomaly = next(
                (a for a in dep_anomalies if a["feature"] == feat_key), None,
            )
            if feat_anomaly is not None:
                severity_mult = 1.0 if "very_" not in feat_anomaly["severity"] else 1.3
                score += weight * severity_mult
                contributing.append({
                    "feature": feat_key,
                    "weight": weight,
                    "contribution": round(weight * severity_mult, 4),
                    "value": feat_anomaly["value"],
                })

        risk_score = round(min(score, 1.0), 4)

        if risk_score >= 0.6:
            risk_level = "high"
        elif risk_score >= 0.3:
            risk_level = "moderate"
        else:
            risk_level = "low"

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "contributing_features": sorted(contributing, key=lambda c: -c["contribution"]),
            "num_anomalies": len(dep_anomalies),
        }

    # ------------------------------------------------------------------
    # Anomaly explanations
    # ------------------------------------------------------------------

    @staticmethod
    def _build_anomaly_details(
        anomalies: List[Dict[str, Any]],
        features: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        """Produce human-readable explanations for each anomaly."""
        explanations: List[Dict[str, str]] = []

        _messages: Dict[str, str] = {
            "jitter_local": "Elevated jitter indicates vocal fold irregularity, a hallmark of vocal tremor.",
            "shimmer_local": "Elevated shimmer reflects amplitude perturbation, suggesting reduced laryngeal control.",
            "hnr_db": "Low harmonics-to-noise ratio indicates increased breathiness or aperiodicity.",
            "pd_speech_rate": "Slower than normal speech rate can signal motor speech impairment.",
            "pd_f0_std": "Reduced pitch variability (monotone) is associated with hypokinetic dysarthria.",
            "pd_pause_ratio": "Excessive pausing relative to speech duration suggests speech planning or motor deficits.",
            "dep_f0_std": "Reduced pitch variability (monotone prosody) is a classic marker of depressive speech.",
            "dep_f0_range": "Narrow pitch range indicates emotional flattening common in depression.",
            "dep_spectral_flatness": "Elevated spectral flatness suggests noisier, less tonal speech quality.",
            "dep_pause_ratio": "Increased pausing ratio may reflect psychomotor retardation.",
            "dep_mean_pause_duration": "Longer average pauses suggest slowed cognitive processing.",
            "dep_speech_rate": "Slower speech rate is associated with depressive psychomotor slowing.",
            "dep_rms_std": "Reduced energy variability indicates flat affect and low vocal expressiveness.",
            "dep_voicing_fraction": "Lower voicing fraction suggests reduced vocal effort.",
        }

        for anomaly in anomalies:
            feat = anomaly["feature"]
            severity = anomaly["severity"]
            explanations.append({
                "feature": feat,
                "condition": anomaly["condition"],
                "severity": severity,
                "value": str(anomaly["value"]),
                "threshold": str(anomaly["threshold"]),
                "explanation": _messages.get(
                    feat,
                    f"Feature '{feat}' is outside normal range (severity: {severity}).",
                ),
            })

        return explanations
