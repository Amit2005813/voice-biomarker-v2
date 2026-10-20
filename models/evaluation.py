"""Model evaluation utilities for Voice Biomarker Platform v2.

Provides functions for computing metrics, generating plots, producing
classification reports, and comparing multiple models.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute a full suite of classification metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Class probability estimates.  Used for ROC-AUC.  If the problem is
        binary the positive-class column (index 1) is used.

    Returns
    -------
    dict
        ``"accuracy"``, ``"f1_macro"``, ``"f1_weighted"``,
        ``"precision_macro"``, ``"recall_macro"``, ``"roc_auc"``.
    """
    n_classes = len(np.unique(y_true))

    roc_auc_val: float = 0.0
    if y_prob is not None:
        try:
            if n_classes == 2 and y_prob.ndim == 2:
                roc_auc_val = float(roc_auc_score(y_true, y_prob[:, 1]))
            elif y_prob.ndim == 2:
                roc_auc_val = float(roc_auc_score(y_true, y_prob, multi_class="ovr"))
        except ValueError as exc:
            logger.warning("ROC-AUC could not be computed: %s", exc)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "roc_auc": roc_auc_val,
    }


# ------------------------------------------------------------------
# Classification report
# ------------------------------------------------------------------

def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: Optional[List[str]] = None,
) -> str:
    """Return a formatted classification-report string."""
    return classification_report(y_true, y_pred, target_names=target_names)


# ------------------------------------------------------------------
# Plot: Confusion Matrix
# ------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    normalize: bool = False,
) -> Path:
    """Save a confusion-matrix heatmap and return the file path."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)

    n = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(n)]

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i,
                f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]:d}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    path = Path(save_path) if save_path else Path("confusion_matrix.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", path)
    return path


# ------------------------------------------------------------------
# Plot: ROC Curve
# ------------------------------------------------------------------

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save ROC curve plot and return the file path.

    For binary problems a single curve is drawn.  For multi-class problems
    one-vs-rest curves are plotted.
    """
    n_classes = len(np.unique(y_true))
    fig, ax = plt.subplots(figsize=(8, 6))

    if n_classes == 2 and y_prob.ndim == 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
        auc = roc_auc_score(y_true, y_prob[:, 1])
        label = class_names[1] if class_names and len(class_names) > 1 else "Positive"
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc:.3f})")
    elif y_prob.ndim == 2:
        from sklearn.preprocessing import label_binarize
        y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
            name = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
            ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = Path(save_path) if save_path else Path("roc_curve.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved to %s", path)
    return path


# ------------------------------------------------------------------
# Plot: Feature Importance
# ------------------------------------------------------------------

def plot_feature_importance(
    importances: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    save_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Save a horizontal bar chart of the top *top_n* features."""
    n_features = len(importances)
    top_n = min(top_n, n_features)

    indices = np.argsort(importances)[-top_n:]
    sorted_imp = importances[indices]

    if feature_names is not None:
        labels = [feature_names[i] for i in indices]
    else:
        labels = [f"Feature {i}" for i in indices]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(labels, sorted_imp, color="#4c72b0", edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    path = Path(save_path) if save_path else Path("feature_importance.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance plot saved to %s", path)
    return path


# ------------------------------------------------------------------
# Model comparison
# ------------------------------------------------------------------

def compare_models(
    results: Dict[str, Dict[str, float]],
    metrics: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
) -> Optional[Path]:
    """Compare multiple models side-by-side with a grouped bar chart.

    Parameters
    ----------
    results : dict
        Mapping ``{model_name: {metric_name: value, ...}}``.
    metrics : list of str, optional
        Subset of metrics to include.  Defaults to all keys in the first
        model's result dict.
    save_path : str or Path, optional
        If provided the chart is saved and the path returned.

    Returns
    -------
    Path or None
    """
    if not results:
        logger.warning("No results to compare.")
        return None

    if metrics is None:
        metrics = list(next(iter(results.values())).keys())

    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)

    data = np.array([[results[m].get(metric, 0.0) for metric in metrics] for m in model_names])

    x = np.arange(n_metrics)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.5), 5))

    for i, name in enumerate(model_names):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, data[i], width, label=name)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path is None:
        save_path = Path("model_comparison.png")
    else:
        save_path = Path(save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Model comparison chart saved to %s", save_path)
    return save_path
