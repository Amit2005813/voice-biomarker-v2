"""Diagnosis screening agent for Voice Biomarker Platform v2."""

from __future__ import annotations

import logging
import joblib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from agents.base_agent import BaseAgent
from config.settings import get_settings

logger = logging.getLogger(__name__)

_MODEL_CACHE: Dict[str, Any] = {}


class DiagnosisAgent(BaseAgent):
    """Run trained models (XGBoost / deep) or rule-based fallback
    to produce diagnosis predictions with confidence and risk levels.

    Parameters
    ----------
    checkpoint_dir : str or Path, optional
        Directory containing saved model checkpoints.
    use_deep_model : bool, optional
        Force deep model usage. ``None`` reads from config.
    """

    def __init__(
        self,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        use_deep_model: Optional[bool] = None,
    ) -> None:
        super().__init__(name="diagnosis")
        settings = get_settings()
        self._checkpoint_dir = Path(
            checkpoint_dir or Path(settings.base_dir) / settings.model.checkpoint_dir
        )
        self._use_deep = use_deep_model if use_deep_model is not None else settings.model.use_deep_model
        self._model_type = settings.model.model_type

        self._parkinsons_model = None
        self._depression_model = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_models_loaded(self) -> None:
        """Lazily load trained model checkpoints."""
        if self._parkinsons_model is not None and self._depression_model is not None:
            return

        pd_path = self._checkpoint_dir / "parkinsons_xgboost.pkl"
        dep_path = self._checkpoint_dir / "depression_xgboost.pkl"

        if not pd_path.exists():
            pd_path = self._checkpoint_dir / "parkinsons" / "model.pkl"
        if not dep_path.exists():
            dep_path = self._checkpoint_dir / "depression" / "model.pkl"

        if self._use_deep:
            pd_path = self._checkpoint_dir / "parkinsons" / "model.pt"
            dep_path = self._checkpoint_dir / "depression" / "model.pt"

        self._parkinsons_model = self._load_model(pd_path, "parkinsons")
        self._depression_model = self._load_model(dep_path, "depression")

    def _load_model(self, path: Path, condition: str) -> Any:
        """Load a single model from disk, returning ``None`` if unavailable."""
        if path.exists():
            cache_key = str(path)
            if cache_key in _MODEL_CACHE:
                self.logger.info("Using cached model for %s", condition)
                return _MODEL_CACHE[cache_key]

            try:
                if self._use_deep:
                    from models.deep_models import Wav2Vec2Classifier
                    model = Wav2Vec2Classifier.load(path)
                else:
                    from models.classical_models import ClassicalClassifier
                    model = ClassicalClassifier.load(path)

                _MODEL_CACHE[cache_key] = model
                self.logger.info("Loaded %s model from %s", condition, path)
                return model
            except Exception as exc:
                self.logger.warning("Failed to load %s model: %s", condition, exc)
        else:
            self.logger.info("No %s model found at %s – using rule-based fallback", condition, path)

        return None

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _execute(
        self,
        features: Dict[str, Any],
        biomarkers: Dict[str, Any],
        audio: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run diagnosis for Parkinson's and Depression.

        Parameters
        ----------
        features : dict
            Scalar acoustic features.
        biomarkers : dict
            Output of :class:`BiomarkerAgent`.
        audio : np.ndarray, optional
            Raw audio waveform. Required for deep models.

        Returns
        -------
        dict
            Keys: ``parkinsons``, ``depression``, ``escalation_needed``.
        """
        self._ensure_models_loaded()

        pd_result = self._predict_condition(
            features, audio, self._parkinsons_model, "parkinsons",
        )
        dep_result = self._predict_condition(
            features, audio, self._depression_model, "depression",
        )

        # Merge biomarker risk scores into final risk when no model is available
        pd_risk = pd_result["risk_score"]
        dep_risk = dep_result["risk_score"]

        # Escalation logic
        escalation_needed = (
            pd_risk >= 0.6 or dep_risk >= 0.6
            or pd_result["confidence"] < 0.4
            or dep_result["confidence"] < 0.4
        )

        recommendations = self._build_recommendations(
            pd_result, dep_result, escalation_needed,
        )

        results = {
            "parkinsons": pd_result,
            "depression": dep_result,
            "escalation_needed": escalation_needed,
            "recommendations": recommendations,
        }

        self.logger.info(
            "Diagnosis: PD risk=%.3f (conf=%.3f, method=%s), "
            "Dep risk=%.3f (conf=%.3f, method=%s), escalation=%s",
            pd_risk, pd_result["confidence"], pd_result["method"],
            dep_risk, dep_result["confidence"], dep_result["method"],
            escalation_needed,
        )
        return results

    # ------------------------------------------------------------------
    # Prediction per condition
    # ------------------------------------------------------------------

    def _predict_condition(
        self,
        features: Dict[str, Any],
        audio: Optional[np.ndarray],
        model: Any,
        condition: str,
    ) -> Dict[str, Any]:
        """Predict a single condition using model or rule-based fallback."""
        if model is not None:
            return self._model_predict(features, audio, model, condition)
        return self._rule_based_predict(features, condition)

    def _model_predict(
        self,
        features: Dict[str, Any],
        audio: Optional[np.ndarray],
        model: Any,
        condition: str,
    ) -> Dict[str, Any]:
        """Use a loaded model to predict."""
        # Classical models work on feature vectors
        is_deep = hasattr(model, "forward") and not hasattr(model, "_pipeline")

        if is_deep:
            if audio is None:
                self.logger.warning("Deep model requires audio; falling back to rule-based")
                return self._rule_based_predict(features, condition)

            predictions = model.predict(audio)
            probabilities = model.predict_proba(audio)
            predicted_class = int(predictions[0])
            confidence = float(np.max(probabilities[0]))
            risk_score = float(probabilities[0][1]) if probabilities.shape[1] >= 2 else confidence
        else:
            vector = self._features_to_vector(features)
            if vector is None:
                return self._rule_based_predict(features, condition)

            predicted_class = int(model.predict(vector)[0])
            probabilities = model.predict_proba(vector)
            confidence = float(np.max(probabilities[0]))
            risk_score = float(probabilities[0][1]) if probabilities.shape[1] >= 2 else confidence

        risk_level = self._risk_level(risk_score)
        label = "positive" if predicted_class == 1 else "negative"

        return {
            "prediction": label,
            "predicted_class": predicted_class,
            "risk_score": round(risk_score, 4),
            "risk_level": risk_level,
            "confidence": round(confidence, 4),
            "method": "model",
            "model_type": type(model).__name__,
        }

    def _features_to_vector(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Convert scalar features to a 2-D numpy array, selecting only features the model was trained on."""
        feat_names_path = self._checkpoint_dir / "feature_names.pkl"
        if feat_names_path.exists():
            try:
                expected_names = joblib.load(feat_names_path)
                vector = [[features.get(name, 0.0) for name in expected_names]]
                return np.array(vector, dtype=np.float32)
            except Exception:
                pass
        scalars = [(k, v) for k, v in sorted(features.items()) if isinstance(v, (int, float))]
        if not scalars:
            return None
        return np.array([[v for _, v in scalars]])

    # ------------------------------------------------------------------
    # Rule-based fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _rule_based_predict(
        features: Dict[str, Any],
        condition: str,
    ) -> Dict[str, Any]:
        """Heuristic prediction when no trained model is available."""
        if condition == "parkinsons":
            indicators = {
                "jitter_local": features.get("jitter_local", 0),
                "shimmer_local": features.get("shimmer_local", 0),
                "hnr_db": features.get("hnr_db", 20),
                "pd_speech_rate": features.get("pd_speech_rate", 4),
                "pd_f0_std": features.get("pd_f0_std", 20),
            }

            score = 0.0
            # Elevated jitter
            if indicators["jitter_local"] > 0.015:
                score += 0.25
            elif indicators["jitter_local"] > 0.010:
                score += 0.12
            # Elevated shimmer
            if indicators["shimmer_local"] > 0.05:
                score += 0.25
            elif indicators["shimmer_local"] > 0.03:
                score += 0.10
            # Low HNR
            if indicators["hnr_db"] < 5.0:
                score += 0.25
            elif indicators["hnr_db"] < 10.0:
                score += 0.12
            # Slow speech rate
            if indicators["pd_speech_rate"] < 2.0:
                score += 0.15
            elif indicators["pd_speech_rate"] < 3.0:
                score += 0.08
            # Low pitch variability
            if indicators["pd_f0_std"] < 5.0:
                score += 0.10
            elif indicators["pd_f0_std"] < 10.0:
                score += 0.05

        else:  # depression
            indicators = {
                "dep_f0_std": features.get("dep_f0_std", 25),
                "dep_f0_range": features.get("dep_f0_range", 50),
                "dep_pause_ratio": features.get("dep_pause_ratio", 0.1),
                "dep_speech_rate": features.get("dep_speech_rate", 4),
                "dep_spectral_flatness": features.get("dep_spectral_flatness", 0.02),
                "dep_rms_std": features.get("dep_rms_std", 0.03),
            }

            score = 0.0
            # Low pitch variability
            if indicators["dep_f0_std"] < 8.0:
                score += 0.20
            elif indicators["dep_f0_std"] < 15.0:
                score += 0.10
            # Narrow pitch range
            if indicators["dep_f0_range"] < 15.0:
                score += 0.18
            elif indicators["dep_f0_range"] < 30.0:
                score += 0.09
            # High pause ratio
            if indicators["dep_pause_ratio"] > 0.40:
                score += 0.20
            elif indicators["dep_pause_ratio"] > 0.25:
                score += 0.10
            # Slow speech
            if indicators["dep_speech_rate"] < 1.5:
                score += 0.15
            elif indicators["dep_speech_rate"] < 2.5:
                score += 0.07
            # Spectral flatness
            if indicators["dep_spectral_flatness"] > 0.25:
                score += 0.15
            elif indicators["dep_spectral_flatness"] > 0.10:
                score += 0.07
            # Energy variability
            if indicators["dep_rms_std"] < 0.005:
                score += 0.12
            elif indicators["dep_rms_std"] < 0.01:
                score += 0.06

        risk_score = round(min(score, 1.0), 4)
        predicted_class = 1 if risk_score >= 0.5 else 0
        confidence = round(max(risk_score, 1 - risk_score), 4)

        return {
            "prediction": "positive" if predicted_class == 1 else "negative",
            "predicted_class": predicted_class,
            "risk_score": risk_score,
            "risk_level": DiagnosisAgent._risk_level(risk_score),
            "confidence": confidence,
            "method": "rule_based",
            "model_type": None,
        }

    @staticmethod
    def _risk_level(score: float) -> str:
        if score >= 0.6:
            return "high"
        if score >= 0.3:
            return "moderate"
        return "low"

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    @staticmethod
    def _build_recommendations(
        pd_result: Dict[str, Any],
        dep_result: Dict[str, Any],
        escalation_needed: bool,
    ) -> List[str]:
        """Generate actionable recommendations based on results."""
        recs: List[str] = []

        if pd_result["risk_level"] == "high":
            recs.append(
                "High Parkinson's risk detected. Recommend specialist neurology referral "
                "for clinical evaluation and confirmatory testing."
            )
        elif pd_result["risk_level"] == "moderate":
            recs.append(
                "Moderate Parkinson's indicators present. Consider follow-up screening "
                "in 3-6 months and monitoring for motor symptoms."
            )

        if dep_result["risk_level"] == "high":
            recs.append(
                "High depression risk detected. Recommend mental health professional "
                "consultation and standardized depression assessment (PHQ-9)."
            )
        elif dep_result["risk_level"] == "moderate":
            recs.append(
                "Moderate depression indicators present. Suggest self-assessment "
                "questionnaire and follow-up if symptoms persist."
            )

        if escalation_needed:
            recs.append(
                "Model confidence is below threshold – results should be reviewed "
                "by a qualified clinician before clinical decisions are made."
            )

        if pd_result["method"] == "rule_based" and dep_result["method"] == "rule_based":
            recs.append(
                "Note: Results are based on rule-based heuristics. "
                "For higher accuracy, train disease-specific models on labeled data."
            )

        if not recs:
            recs.append(
                "No significant voice biomarker anomalies detected. "
                "Continue routine monitoring as appropriate."
            )

        return recs
