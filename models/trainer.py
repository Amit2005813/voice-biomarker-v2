"""Training orchestration for Voice Biomarker Platform v2.

Provides ``ModelTrainer`` which coordinates classical and deep model
training, hyper-parameter search, evaluation, and result persistence.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.settings import TrainingConfig
from models.classical_models import ClassicalClassifier, SUPPORTED_MODEL_TYPES
from models.evaluation import (
    compute_metrics,
    generate_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """High-level training orchestrator.

    Wraps classical and deep model training with cross-validation,
    hyper-parameter search, result persistence, and plotting.

    Parameters
    ----------
    config : TrainingConfig
        Training hyper-parameters.
    output_dir : str or Path, optional
        Directory to store artefacts.  Defaults to ``"models/checkpoints"``.
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        output_dir: Union[str, Path] = "models/checkpoints",
    ) -> None:
        self.config = config or TrainingConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict[str, Any] = {}
        self._training_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Classical model training
    # ------------------------------------------------------------------

    def train_classical(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "xgboost",
        cv_folds: int = 5,
        **model_kwargs: Any,
    ) -> Dict[str, Any]:
        """Train a classical model with stratified cross-validation.

        Parameters
        ----------
        X : array-like
        y : array-like
        model_type : str
        cv_folds : int
        **model_kwargs
            Forwarded to ``ClassicalClassifier``.

        Returns
        -------
        dict
            Training metrics including ``cv_mean``, ``cv_std`` and
            per-metric evaluation.
        """
        logger.info("Training classical model: %s", model_type)

        classifier = ClassicalClassifier(
            model_type=model_type, n_classes=len(np.unique(y)), **model_kwargs,
        )
        train_metrics = classifier.fit(X, y, cv_folds=cv_folds)
        eval_metrics = classifier.evaluate(X, y)

        report_text = generate_classification_report(y, classifier.predict(X))

        result: Dict[str, Any] = {
            "model_type": model_type,
            "train_metrics": train_metrics,
            "eval_metrics": eval_metrics,
            "classification_report": report_text,
        }

        self._training_history.append(result)
        self.results[model_type] = result

        logger.info("Model %s trained – CV accuracy: %.4f",
                     model_type, train_metrics["cv_mean"])

        return result

    # ------------------------------------------------------------------
    # Deep model training
    # ------------------------------------------------------------------

    def train_deep(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Any,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        patience: Optional[int] = None,
        device: Optional[str] = None,
        class_weights: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Train a Wav2Vec2Classifier (or compatible deep model).

        Parameters
        ----------
        model : Wav2Vec2Classifier
        train_loader : DataLoader
        val_loader : DataLoader
        epochs, lr, patience : overrides for config defaults
        device : str, optional
        class_weights : Tensor, optional

        Returns
        -------
        dict
            Training history and timing info.
        """
        epochs = epochs or self.config.epochs
        lr = lr or self.config.learning_rate
        patience = patience or self.config.early_stopping_patience

        logger.info("Training deep model for up to %d epochs", epochs)
        t0 = time.time()

        history = model.fit(
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            patience=patience,
            device=device,
            class_weights=class_weights,
        )

        elapsed = time.time() - t0
        history["elapsed_seconds"] = elapsed

        self._training_history.append({"model_type": "deep", "history": history})
        self.results["deep"] = history

        logger.info("Deep model training complete in %.1fs", elapsed)
        return history

    # ------------------------------------------------------------------
    # Hyper-parameter search
    # ------------------------------------------------------------------

    def hyperparameter_search(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List[Any]],
        model_type: str = "xgboost",
        search_type: str = "grid",
        cv_folds: int = 5,
        n_iter: int = 50,
        scoring: str = "f1_macro",
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        """Run grid or randomised search over *param_grid*.

        Parameters
        ----------
        X, y : training data
        param_grid : dict
            Keys are prefixed with ``model__`` to reach the estimator inside
            the sklearn Pipeline (e.g. ``{"model__n_estimators": [100, 200]}``).
        model_type : str
        search_type : ``"grid"`` or ``"random"``
        cv_folds : int
        n_iter : int
            Number of random iterations (only used when *search_type* == ``"random"``).
        scoring : str
        n_jobs : int

        Returns
        -------
        dict
            ``"best_params"``, ``"best_score"``, ``"cv_results"``
        """
        estimator_cls = SUPPORTED_MODEL_TYPES.get(model_type)
        if estimator_cls is None:
            raise ValueError(f"Unsupported model_type: {model_type}")

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", estimator_cls()),
        ])

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        if search_type == "grid":
            searcher: Any = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring=scoring,
                n_jobs=n_jobs, refit=True, return_train_score=False,
            )
        elif search_type == "random":
            searcher = RandomizedSearchCV(
                pipeline, param_grid, n_iter=n_iter, cv=cv,
                scoring=scoring, n_jobs=n_jobs, refit=True,
                return_train_score=False, random_state=42,
            )
        else:
            raise ValueError(f"Unknown search_type: {search_type!r}")

        logger.info("Starting %s search with %d folds", search_type, cv_folds)
        searcher.fit(X, y)

        result: Dict[str, Any] = {
            "best_params": searcher.best_params_,
            "best_score": float(searcher.best_score_),
            "cv_results": {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in searcher.cv_results_.items()
            },
        }

        logger.info("Best %s score: %.4f", scoring, result["best_score"])
        return result

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_model(
        self,
        model: Union[ClassicalClassifier, Any],
        X: np.ndarray,
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """Comprehensive evaluation with metrics and plots.

        Returns
        -------
        dict
            ``"metrics"``, ``"confusion_matrix_path"``, ``"roc_curve_path"``,
            ``"classification_report"``.
        """
        if isinstance(model, ClassicalClassifier):
            y_pred = model.predict(X)
            y_prob = model.predict_proba(X)
        else:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model.eval()
            with torch.no_grad():
                logits = model(torch.tensor(X, dtype=torch.float32).to(device))
                y_prob = torch.softmax(logits, dim=1).cpu().numpy()
                y_pred = y_prob.argmax(axis=1)

        metrics = compute_metrics(y, y_pred, y_prob)
        report = generate_classification_report(y, y_pred)

        cm_path = plot_confusion_matrix(y, y_pred, save_path=self.output_dir / "confusion_matrix.png")
        roc_path = plot_roc_curve(y, y_prob, save_path=self.output_dir / "roc_curve.png")

        return {
            "metrics": metrics,
            "confusion_matrix_path": str(cm_path),
            "roc_curve_path": str(roc_path),
            "classification_report": report,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_training_results(self) -> Path:
        """Serialise metrics and training history to disk.

        Returns
        -------
        Path
            Path to the saved JSON file.
        """
        out_path = self.output_dir / "training_results.json"

        serialisable = json.loads(json.dumps(self.results, default=str))

        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2)

        logger.info("Training results saved to %s", out_path)
        return out_path

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_training_history(
        self, history: Optional[Dict[str, List[float]]] = None, save_path: Optional[Path] = None,
    ) -> Path:
        """Plot loss and accuracy curves for a deep training run.

        Parameters
        ----------
        history : dict, optional
            If *None* the most recent deep training history is used.
        save_path : Path, optional
            Defaults to ``output_dir / "training_history.png"``.

        Returns
        -------
        Path
        """
        if history is None:
            for entry in reversed(self._training_history):
                if "history" in entry:
                    history = entry["history"]
                    break
            if history is None:
                raise ValueError("No training history available to plot.")

        if save_path is None:
            save_path = self.output_dir / "training_history.png"

        epochs = range(1, len(history["train_loss"]) + 1)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(epochs, history["train_loss"], label="Train Loss")
        axes[0].plot(epochs, history["val_loss"], label="Val Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Loss Curve")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, history["train_acc"], label="Train Acc")
        axes[1].plot(epochs, history["val_acc"], label="Val Acc")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Accuracy Curve")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.tight_layout()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Training history plot saved to %s", save_path)
        return save_path
