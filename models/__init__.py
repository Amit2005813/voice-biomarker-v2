"""ML models module for Voice Biomarker Platform v2.

Provides classical ML classifiers, deep learning models based on Wav2Vec2,
and orchestration utilities for training and evaluation.
"""

from models.classical_models import ClassicalClassifier
from models.deep_models import Wav2Vec2Classifier
from models.evaluation import (
    compute_metrics,
    compare_models,
    generate_classification_report,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)
from models.trainer import ModelTrainer

__all__ = [
    "ClassicalClassifier",
    "Wav2Vec2Classifier",
    "ModelTrainer",
    "compute_metrics",
    "compare_models",
    "generate_classification_report",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_roc_curve",
]
