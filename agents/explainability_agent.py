"""Explainability agent for Voice Biomarker Platform v2.

Uses SHAP and LIME to produce feature-level explanations for
model predictions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from agents.base_agent import BaseAgent
from config.settings import get_settings

logger = logging.getLogger(__name__)

_SHAP_AVAILABLE = False
try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    pass

_LIME_AVAILABLE = False
try:
    from lime.lime_tabular import LimeTabularExplainer
    _LIME_AVAILABLE = True
except ImportError:
    pass


class ExplainabilityAgent(BaseAgent):
    """Generate model-agnostic and model-specific explanations.

    Supports:
    * **SHAP** (TreeExplainer for tree-based models, KernelExplainer fallback).
    * **LIME** (tabular explanations).
    * **Feature importance** rankings from the model itself.
    * **Plain-language** summaries of contributing features.

    Parameters
    ----------
    num_top_features : int, optional
        Number of top features to include in explanations.
    use_shap : bool, optional
        Enable SHAP explanations.
    use_lime : bool, optional
        Enable LIME explanations.
    """

    def __init__(
        self,
        num_top_features: Optional[int] = None,
        use_shap: Optional[bool] = None,
        use_lime: Optional[bool] = None,
    ) -> None:
        super().__init__(name="explainability")
        settings = get_settings()
        self._num_top = num_top_features or settings.explainability.num_features
        self._use_shap = use_shap if use_shap is not None else settings.explainability.use_shap
        self._use_lime = use_lime if use_lime is not None else settings.explainability.use_lime

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _execute(
        self,
        features: Dict[str, Any],
        feature_names: List[str],
        model: Any = None,
        diagnosis: Optional[Dict[str, Any]] = None,
        background_data: Optional[np.ndarray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate explanations for a prediction.

        Parameters
        ----------
        features : dict
            Scalar feature dictionary.
        feature_names : list of str
            Ordered feature names matching the feature vector.
        model : fitted model, optional
            The trained model to explain. If ``None`` only heuristic
            explanations are returned.
        diagnosis : dict, optional
            Output of :class:`DiagnosisAgent` for context.
        background_data : np.ndarray, optional
            Background dataset for SHAP (shape ``(n_samples, n_features)``).

        Returns
        -------
        dict
            Keys: ``shap_explanation``, ``lime_explanation``,
            ``feature_importance``, ``top_features``,
            ``plain_language_explanation``.
        """
        vector = self._features_to_vector(features, feature_names)
        if vector is None or len(feature_names) == 0:
            return self._empty_explanation()

        results: Dict[str, Any] = {}

        # 1. SHAP
        shap_values = None
        if self._use_shap and _SHAP_AVAILABLE and model is not None:
            shap_values = self._compute_shap(model, vector, feature_names, background_data)
            results["shap_explanation"] = shap_values
        else:
            results["shap_explanation"] = None

        # 2. LIME
        lime_values = None
        if self._use_lime and _LIME_AVAILABLE and model is not None:
            lime_values = self._compute_lime(model, vector, feature_names, background_data)
            results["lime_explanation"] = lime_values
        else:
            results["lime_explanation"] = None

        # 3. Model feature importances
        importance = self._get_model_importance(model, feature_names)
        results["feature_importance"] = importance

        # 4. Aggregate top features
        top = self._rank_top_features(
            vector, feature_names, shap_values, lime_values, importance,
        )
        results["top_features"] = top

        # 5. Plain-language explanation
        results["plain_language_explanation"] = self._generate_plain_language(
            top, diagnosis,
        )

        self.logger.info(
            "Explanation generated: %d top features, shap=%s, lime=%s",
            len(top),
            shap_values is not None,
            lime_values is not None,
        )
        return results

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def _compute_shap(
        self,
        model: Any,
        vector: np.ndarray,
        feature_names: List[str],
        background: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Compute SHAP values for the given prediction."""
        try:
            is_tree = self._is_tree_model(model)

            if is_tree:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(vector)
            else:
                bg = background if background is not None else vector
                background_summary = shap.sample(bg, min(100, len(bg)))
                explainer = shap.KernelExplainer(
                    self._model_predict_fn(model), background_summary,
                )
                shap_vals = explainer.shap_values(vector, nsamples=100)

            # Handle multi-output (list of arrays)
            if isinstance(shap_vals, list):
                vals = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
            else:
                vals = shap_vals

            vals_flat = np.asarray(vals).flatten()
            if len(vals_flat) != len(feature_names):
                self.logger.warning(
                    "SHAP output length %d != feature count %d", len(vals_flat), len(feature_names),
                )
                vals_flat = np.zeros(len(feature_names))

            ranking = sorted(
                zip(feature_names, vals_flat.tolist()),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[: self._num_top]

            return {
                "values": vals_flat.tolist(),
                "top_contributors": [
                    {"feature": n, "shap_value": round(float(v), 6)}
                    for n, v in ranking
                ],
            }

        except Exception as exc:
            self.logger.warning("SHAP computation failed: %s", exc)
            return {"values": None, "top_contributors": [], "error": str(exc)}

    # ------------------------------------------------------------------
    # LIME
    # ------------------------------------------------------------------

    def _compute_lime(
        self,
        model: Any,
        vector: np.ndarray,
        feature_names: List[str],
        background: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Compute LIME explanation for the given prediction."""
        try:
            training_data = background if background is not None else vector

            explainer = LimeTabularExplainer(
                training_data=training_data,
                feature_names=feature_names,
                mode="classification",
                discretize_continuous=True,
            )

            predict_fn = self._model_predict_fn(model)

            explanation = explainer.explain_instance(
                vector.flatten(),
                predict_fn,
                num_features=min(self._num_top, len(feature_names)),
            )

            raw = explanation.as_list()
            return {
                "feature_contributions": [
                    {"feature": str(feat), "weight": round(float(w), 6)}
                    for feat, w in raw
                ],
                "score": round(float(explanation.score), 4) if hasattr(explanation, "score") else None,
            }

        except Exception as exc:
            self.logger.warning("LIME computation failed: %s", exc)
            return {"feature_contributions": [], "error": str(exc)}

    # ------------------------------------------------------------------
    # Model importance
    # ------------------------------------------------------------------

    @staticmethod
    def _get_model_importance(
        model: Any, feature_names: List[str],
    ) -> Optional[List[Dict[str, Any]]]:
        """Extract native feature importances from tree-based models."""
        if model is None:
            return None

        try:
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif hasattr(model, "get_booster"):
                importances = model.get_booster().get_score(
                    importance_type="gain"
                )
                imp_array = np.zeros(len(feature_names))
                for k, v in importances.items():
                    idx = int(k.replace("f", "")) if k.startswith("f") else -1
                    if 0 <= idx < len(feature_names):
                        imp_array[idx] = v
                total = imp_array.sum()
                importances = imp_array / total if total > 0 else imp_array
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_).flatten()
            else:
                return None

            if len(importances) != len(feature_names):
                return None

            ranking = sorted(
                zip(feature_names, importances.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )[: len(feature_names)]

            total_imp = sum(v for _, v in ranking) or 1.0
            return [
                {
                    "feature": n,
                    "importance": round(float(v), 6),
                    "relative_pct": round(float(v / total_imp) * 100, 2),
                }
                for n, v in ranking
            ]

        except Exception as exc:
            logger.debug("Could not extract feature importances: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Top feature ranking
    # ------------------------------------------------------------------

    def _rank_top_features(
        self,
        vector: np.ndarray,
        feature_names: List[str],
        shap_result: Optional[Dict[str, Any]],
        lime_result: Optional[Dict[str, Any]],
        importance: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Merge multiple explanation signals into a unified ranking."""
        scores: Dict[str, float] = {n: 0.0 for n in feature_names}
        methods_used = 0

        # SHAP contribution magnitude
        if shap_result and shap_result.get("values") is not None:
            vals = np.abs(np.array(shap_result["values"]))
            if len(vals) == len(feature_names):
                max_v = vals.max() or 1.0
                for i, name in enumerate(feature_names):
                    scores[name] += vals[i] / max_v
                methods_used += 1

        # LIME weight magnitude
        if lime_result and lime_result.get("feature_contributions"):
            max_w = max(abs(c["weight"]) for c in lime_result["feature_contributions"]) or 1.0
            for entry in lime_result["feature_contributions"]:
                fname = entry["feature"]
                if fname in scores:
                    scores[fname] += abs(entry["weight"]) / max_w
            methods_used += 1

        # Native importance
        if importance:
            max_i = max(e["importance"] for e in importance) or 1.0
            for entry in importance:
                scores[entry["feature"]] += entry["importance"] / max_i
            methods_used += 1

        if methods_used == 0:
            return []

        # Average across methods
        ranked = sorted(
            [(n, s / methods_used) for n, s in scores.items()],
            key=lambda x: x[1],
            reverse=True,
        )[: self._num_top]

        return [
            {"feature": n, "relevance_score": round(s, 4), "rank": i + 1}
            for i, (n, s) in enumerate(ranked)
        ]

    # ------------------------------------------------------------------
    # Plain language
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_plain_language(
        top_features: List[Dict[str, Any]],
        diagnosis: Optional[Dict[str, Any]],
    ) -> str:
        """Create a human-readable explanation paragraph."""
        if not top_features:
            return "No significant feature contributions identified."

        lines: List[str] = []

        # Opening
        if diagnosis:
            pd = diagnosis.get("parkinsons", {})
            dep = diagnosis.get("depression", {})
            pd_risk = pd.get("risk_level", "unknown")
            dep_risk = dep.get("risk_level", "unknown")
            lines.append(
                f"Analysis detected {pd_risk} risk for Parkinson's indicators "
                f"and {dep_risk} risk for depression indicators."
            )

        lines.append("The following vocal features most influenced the assessment:")

        for feat in top_features[:5]:
            name = feat["feature"]
            score = feat["relevance_score"]
            readable = name.replace("_", " ").replace("pd ", "parkinsons ").replace("dep ", "depression ")
            lines.append(
                f"- {readable.title()} (relevance: {score:.0%})"
            )

        # Context
        lines.append(
            "\nThese findings reflect acoustic patterns in the voice recording "
            "and should be interpreted by a qualified healthcare professional."
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _features_to_vector(
        features: Dict[str, Any], feature_names: List[str],
    ) -> Optional[np.ndarray]:
        """Build a numeric vector from feature dict + ordered names."""
        try:
            vals = [float(features.get(n, 0.0)) for n in feature_names]
            return np.array([vals])
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_tree_model(model: Any) -> bool:
        """Check if model is tree-based (supports TreeExplainer)."""
        if model is None:
            return False
        if hasattr(model, "get_booster"):  # XGBoost
            return True
        if hasattr(model, "estimators_"):  # sklearn ensemble
            return True
        if type(model).__name__ in ("XGBClassifier", "RandomForestClassifier",
                                     "GradientBoostingClassifier"):
            return True
        return False

    def _model_predict_fn(self, model: Any):
        """Return a callable suitable for SHAP/LIME."""
        def predict(X: np.ndarray) -> np.ndarray:
            try:
                proba = model.predict_proba(X)
                return proba
            except Exception:
                preds = model.predict(X)
                if preds.ndim == 1:
                    result = np.zeros((len(preds), 2))
                    result[np.arange(len(preds)), preds.astype(int)] = 1.0
                    return result
                return preds
        return predict

    def _empty_explanation(self) -> Dict[str, Any]:
        return {
            "shap_explanation": None,
            "lime_explanation": None,
            "feature_importance": None,
            "top_features": [],
            "plain_language_explanation": "Insufficient data for explanation generation.",
        }
