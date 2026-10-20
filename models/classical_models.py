"""Classical ML classifiers for Voice Biomarker Platform v2.

Wraps scikit-learn and XGBoost models in a unified interface with
cross-validation, feature importance, and full metric evaluation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)

ModelType = str
_XGBOOST_AVAILABLE = False
try:
    from xgboost import XGBClassifier

    _XGBOOST_AVAILABLE = True
except ImportError:
    pass

SUPPORTED_MODEL_TYPES: Dict[str, type] = {
    "random_forest": RandomForestClassifier,
    "gradient_boosting": GradientBoostingClassifier,
    "svm": SVC,
}
if _XGBOOST_AVAILABLE:
    SUPPORTED_MODEL_TYPES["xgboost"] = XGBClassifier


class ClassicalClassifier:
    """Unified wrapper around classical ML classifiers.

    Supports XGBoost, SVM, Random Forest, and Gradient Boosting. Every model
    is wrapped in a ``sklearn.pipeline.Pipeline`` with a ``StandardScaler``.

    Parameters
    ----------
    model_type : str
        One of the keys in ``SUPPORTED_MODEL_TYPES`` (default ``"xgboost"``).
    n_classes : int
        Number of target classes.  When *n_classes* == 2 the positive-class
        probabilities are used for ROC-AUC.
    **model_kwargs
        Extra keyword arguments forwarded to the underlying estimator.
    """

    def __init__(
        self,
        model_type: ModelType = "xgboost",
        n_classes: int = 2,
        **model_kwargs: Any,
    ) -> None:
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type={model_type!r}. "
                f"Choose from: {list(SUPPORTED_MODEL_TYPES)}"
            )

        self._model_type = model_type
        self._n_classes = n_classes
        self._model_kwargs = model_kwargs
        self._pipeline: Optional[Pipeline] = None
        self._classes: Optional[np.ndarray] = None

        self._build_pipeline()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_pipeline(self) -> None:
        """Instantiate the sklearn Pipeline with scaler + estimator."""
        estimator_cls = SUPPORTED_MODEL_TYPES[self._model_type]
        params = dict(self._model_kwargs)

        if self._model_type == "xgboost" and _XGBOOST_AVAILABLE:
            params.setdefault("eval_metric", "logloss")
            params.setdefault("use_label_encoder", False)
        if self._model_type == "svm":
            params.setdefault("probability", True)

        self._pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", estimator_cls(**params)),
            ]
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None and self._pipeline.classes_ is not None

    @property
    def classes_(self) -> np.ndarray:
        if self._classes is None:
            raise RuntimeError("Model has not been fitted yet.")
        return self._classes

    @property
    def feature_importances_(self) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted yet.")
        model = self._pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            return model.feature_importances_
        if hasattr(model, "coef_"):
            return np.abs(model.coef_).flatten()
        raise AttributeError(
            f"Estimator {type(model).__name__} does not expose feature importances."
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Fit the pipeline with optional stratified cross-validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        eval_set : tuple(X_val, y_val), optional
            Used by XGBoost for early-stopping evaluation.
        cv_folds : int
            Number of stratified folds for cross-validation (default 5).

        Returns
        -------
        dict
            Cross-validation scores keyed ``"cv_mean"``, ``"cv_std"``,
            ``"cv_scores"`` and per-fold accuracy values.
        """
        self._classes = np.unique(y)

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        if self._model_type == "xgboost" and _XGBOOST_AVAILABLE and eval_set is not None:
            X_val, y_val = eval_set
            xgb_model = self._pipeline.named_steps["model"]
            xgb_model.fit(
                self._pipeline.named_steps["scaler"].fit_transform(X),
                y,
                eval_set=[(
                    self._pipeline.named_steps["scaler"].transform(X_val),
                    y_val,
                )],
                verbose=False,
            )
        else:
            cv_scores = cross_val_score(
                self._pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1,
            )

        if self._model_type == "xgboost" and _XGBOOST_AVAILABLE and eval_set is not None:
            cv_scores = cross_val_score(
                self._pipeline, X, y, cv=cv, scoring="accuracy", n_jobs=-1,
            )

        self._pipeline.fit(X, y)

        metrics: Dict[str, Any] = {
            "cv_mean": float(np.mean(cv_scores)),
            "cv_std": float(np.std(cv_scores)),
            "cv_scores": cv_scores.tolist(),
        }
        logger.info(
            "Training complete – CV accuracy %.4f ± %.4f",
            metrics["cv_mean"],
            metrics["cv_std"],
        )
        return metrics

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probability estimates."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return self._pipeline.predict_proba(X)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute a full suite of classification metrics.

        Returns
        -------
        dict
            Keys: ``accuracy``, ``f1_macro``, ``f1_weighted``,
            ``precision_macro``, ``recall_macro``, ``roc_auc``.
        """
        y_pred = self.predict(X)
        y_prob = self.predict_proba(X)

        roc_auc_val: float = 0.0
        if self._n_classes == 2:
            roc_auc_val = float(roc_auc_score(y, y_prob[:, 1]))
        else:
            try:
                roc_auc_val = float(roc_auc_score(y, y_prob, multi_class="ovr"))
            except ValueError:
                roc_auc_val = 0.0

        return {
            "accuracy": float(accuracy_score(y, y_pred)),
            "f1_macro": float(f1_score(y, y_pred, average="macro")),
            "f1_weighted": float(f1_score(y, y_pred, average="weighted")),
            "precision_macro": float(precision_score(y, y_pred, average="macro")),
            "recall_macro": float(recall_score(y, y_pred, average="macro")),
            "roc_auc": roc_auc_val,
        }

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance array (sorted descending)."""
        return self.feature_importances_

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Serialise the full pipeline to *path* via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self._pipeline,
                "model_type": self._model_type,
                "n_classes": self._n_classes,
                "model_kwargs": self._model_kwargs,
                "classes": self._classes,
            },
            path,
        )
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> ClassicalClassifier:
        """Restore a previously saved model from *path*."""
        data = joblib.load(path)
        instance = cls(
            model_type=data["model_type"],
            n_classes=data["n_classes"],
            **data["model_kwargs"],
        )
        instance._pipeline = data["pipeline"]
        instance._classes = data["classes"]
        return instance
