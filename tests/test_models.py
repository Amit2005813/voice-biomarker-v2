"""Tests for models module: ClassicalClassifier, Wav2Vec2Classifier, ModelTrainer, evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from models.classical_models import ClassicalClassifier, SUPPORTED_MODEL_TYPES
from models.evaluation import (
    compute_metrics,
    compare_models,
    generate_classification_report,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)


# ======================================================================
# ClassicalClassifier
# ======================================================================

class TestClassicalClassifierInit:
    """Test ClassicalClassifier initialization."""

    def test_init_xgboost(self) -> None:
        c = ClassicalClassifier(model_type="xgboost")
        assert c._model_type == "xgboost"
        assert c._n_classes == 2
        assert c._pipeline is not None

    def test_init_svm(self) -> None:
        c = ClassicalClassifier(model_type="svm")
        assert c._model_type == "svm"

    def test_init_random_forest(self) -> None:
        c = ClassicalClassifier(model_type="random_forest")
        assert c._model_type == "random_forest"

    def test_init_gradient_boosting(self) -> None:
        c = ClassicalClassifier(model_type="gradient_boosting")
        assert c._model_type == "gradient_boosting"

    def test_init_invalid_type(self) -> None:
        with pytest.raises(ValueError, match="Unsupported model_type"):
            ClassicalClassifier(model_type="invalid_model")

    def test_supported_model_types(self) -> None:
        assert "random_forest" in SUPPORTED_MODEL_TYPES
        assert "svm" in SUPPORTED_MODEL_TYPES
        assert "gradient_boosting" in SUPPORTED_MODEL_TYPES


class TestClassicalClassifierFit:
    """Test ClassicalClassifier training."""

    @pytest.mark.parametrize("model_type", ["random_forest", "svm", "gradient_boosting"])
    def test_fit_returns_metrics(
        self, sample_features_2d: np.ndarray, sample_labels_2d: np.ndarray, model_type: str,
    ) -> None:
        c = ClassicalClassifier(model_type=model_type, n_classes=2)
        metrics = c.fit(sample_features_2d, sample_labels_2d, cv_folds=3)
        assert "cv_mean" in metrics
        assert "cv_std" in metrics
        assert "cv_scores" in metrics
        assert c.is_fitted

    def test_fit_xgboost(self, sample_features_2d: np.ndarray, sample_labels_2d: np.ndarray) -> None:
        c = ClassicalClassifier(model_type="xgboost", n_classes=2)
        metrics = c.fit(sample_features_2d, sample_labels_2d, cv_folds=3)
        assert "cv_mean" in metrics
        assert c.is_fitted


class TestClassicalClassifierPredict:
    """Test ClassicalClassifier prediction."""

    def test_predict(
        self, sample_features_2d: np.ndarray, sample_labels_2d: np.ndarray,
    ) -> None:
        c = ClassicalClassifier(model_type="random_forest", n_classes=2)
        c.fit(sample_features_2d, sample_labels_2d, cv_folds=3)
        preds = c.predict(sample_features_2d)
        assert preds.ndim == 1
        assert len(preds) == len(sample_labels_2d)

    def test_predict_proba(
        self, sample_features_2d: np.ndarray, sample_labels_2d: np.ndarray,
    ) -> None:
        c = ClassicalClassifier(model_type="random_forest", n_classes=2)
        c.fit(sample_features_2d, sample_labels_2d, cv_folds=3)
        proba = c.predict_proba(sample_features_2d)
        assert proba.ndim == 2
        assert proba.shape[1] == 2
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_unfitted(self) -> None:
        c = ClassicalClassifier(model_type="random_forest")
        with pytest.raises((RuntimeError, AttributeError)):
            c.predict(np.zeros((1, 10)))

    def test_predict_proba_unfitted(self) -> None:
        c = ClassicalClassifier(model_type="random_forest")
        with pytest.raises((RuntimeError, AttributeError)):
            c.predict_proba(np.zeros((1, 10)))


class TestClassicalClassifierEvaluate:
    """Test ClassicalClassifier evaluation."""

    def test_evaluate(
        self, sample_features_2d: np.ndarray, sample_labels_2d: np.ndarray,
    ) -> None:
        c = ClassicalClassifier(model_type="random_forest", n_classes=2)
        c.fit(sample_features_2d, sample_labels_2d, cv_folds=3)
        metrics = c.evaluate(sample_features_2d, sample_labels_2d)
        assert "accuracy" in metrics
        assert "f1_macro" in metrics
        assert "roc_auc" in metrics
        assert 0 <= metrics["accuracy"] <= 1


class TestClassicalClassifierSaveLoad:
    """Test model persistence."""

    def test_save_and_load(
        self, sample_features_2d: np.ndarray, sample_labels_2d: np.ndarray, tmp_path: Path,
    ) -> None:
        c = ClassicalClassifier(model_type="random_forest", n_classes=2)
        c.fit(sample_features_2d, sample_labels_2d, cv_folds=3)

        model_path = tmp_path / "model.pkl"
        c.save(model_path)
        assert model_path.exists()

        loaded = ClassicalClassifier.load(model_path)
        assert loaded.is_fitted
        assert loaded._model_type == "random_forest"

        preds_orig = c.predict(sample_features_2d)
        preds_loaded = loaded.predict(sample_features_2d)
        np.testing.assert_array_equal(preds_orig, preds_loaded)


class TestClassicalClassifierFeatureImportance:
    """Test feature importance extraction."""

    def test_feature_importance_random_forest(
        self, sample_features_2d: np.ndarray, sample_labels_2d: np.ndarray,
    ) -> None:
        c = ClassicalClassifier(model_type="random_forest", n_classes=2)
        c.fit(sample_features_2d, sample_labels_2d, cv_folds=3)
        imp = c.feature_importances_
        assert len(imp) == sample_features_2d.shape[1]

    def test_feature_importance_unfitted(self) -> None:
        c = ClassicalClassifier(model_type="random_forest")
        with pytest.raises((RuntimeError, AttributeError)):
            _ = c.feature_importances_


# ======================================================================
# Wav2Vec2Classifier (skipped if torch/transformers not available)
# ======================================================================

class TestWav2Vec2Classifier:
    """Test Wav2Vec2 deep learning classifier."""

    @pytest.fixture(autouse=True)
    def _skip_if_no_torch(self):
        pytest.importorskip("torch", reason="PyTorch not installed")
        pytest.importorskip("transformers", reason="transformers not installed")

    def test_init(self) -> None:
        from models.deep_models import Wav2Vec2Classifier
        m = Wav2Vec2Classifier(num_classes=2, freeze_base=True)
        assert m._num_classes == 2
        assert m._freeze_base is True

    def test_forward(self) -> None:
        import torch
        from models.deep_models import Wav2Vec2Classifier
        m = Wav2Vec2Classifier(num_classes=2, freeze_base=True)
        x = torch.randn(2, 16000)
        logits = m(x)
        assert logits.shape == (2, 2)

    def test_predict(self) -> None:
        import torch
        from models.deep_models import Wav2Vec2Classifier
        m = Wav2Vec2Classifier(num_classes=2, freeze_base=True)
        audio = np.random.randn(16000).astype(np.float32)
        preds = m.predict(audio)
        assert preds.ndim == 1
        assert len(preds) == 1

    def test_predict_proba(self) -> None:
        import torch
        from models.deep_models import Wav2Vec2Classifier
        m = Wav2Vec2Classifier(num_classes=2, freeze_base=True)
        audio = np.random.randn(16000).astype(np.float32)
        proba = m.predict_proba(audio)
        assert proba.shape == (1, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_save_and_load(self, tmp_path: Path) -> None:
        import torch
        from models.deep_models import Wav2Vec2Classifier
        m = Wav2Vec2Classifier(num_classes=2, freeze_base=True)
        path = tmp_path / "model.pt"
        m.save(path)
        assert path.exists()

        loaded = Wav2Vec2Classifier.load(path)
        assert loaded._num_classes == 2

        audio = np.random.randn(16000).astype(np.float32)
        preds_orig = m.predict(audio)
        preds_loaded = loaded.predict(audio)
        np.testing.assert_array_equal(preds_orig, preds_loaded)

    def test_unfreeze_base(self) -> None:
        import torch
        from models.deep_models import Wav2Vec2Classifier
        m = Wav2Vec2Classifier(num_classes=2, freeze_base=True)
        for p in m.wav2vec2.parameters():
            assert not p.requires_grad

        m.unfreeze_base(num_layers=2)
        layers = m.wav2vec2.encoder.layer
        total = len(layers)
        for i, layer in enumerate(layers):
            for p in layer.parameters():
                if i >= total - 2:
                    assert p.requires_grad

    def test_unfreeze_all(self) -> None:
        import torch
        from models.deep_models import Wav2Vec2Classifier
        m = Wav2Vec2Classifier(num_classes=2, freeze_base=True)
        m.unfreeze_base()
        for p in m.wav2vec2.parameters():
            assert p.requires_grad


# ======================================================================
# Evaluation utilities
# ======================================================================

class TestComputeMetrics:
    """Test compute_metrics function."""

    def test_binary_perfect(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1_macro"] == 1.0
        assert metrics["roc_auc"] == 1.0

    def test_binary_imperfect(self) -> None:
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 0, 1])
        y_prob = np.array([[0.8, 0.2], [0.4, 0.6], [0.3, 0.7], [0.6, 0.4], [0.2, 0.8]])
        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

    def test_no_prob(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["roc_auc"] == 0.0

    def test_multiclass(self) -> None:
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 1, 0, 2])
        y_prob = np.eye(3)[y_pred]
        metrics = compute_metrics(y_true, y_pred, y_prob)
        assert 0 <= metrics["accuracy"] <= 1


class TestGenerateClassificationReport:
    """Test classification report generation."""

    def test_report(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        report = generate_classification_report(y_true, y_pred)
        assert isinstance(report, str)
        assert "precision" in report.lower() or "Precision" in report

    def test_report_with_names(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        report = generate_classification_report(y_true, y_pred, target_names=["Healthy", "PD"])
        assert "Healthy" in report
        assert "PD" in report


class TestPlotConfusionMatrix:
    """Test confusion matrix plotting."""

    def test_plot(self, tmp_path: Path) -> None:
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1])
        path = plot_confusion_matrix(y_true, y_pred, save_path=tmp_path / "cm.png")
        assert path.exists()

    def test_plot_normalized(self, tmp_path: Path) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])
        path = plot_confusion_matrix(y_true, y_pred, save_path=tmp_path / "cm_norm.png", normalize=True)
        assert path.exists()


class TestPlotRocCurve:
    """Test ROC curve plotting."""

    def test_plot_binary(self, tmp_path: Path) -> None:
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8], [0.9, 0.1], [0.4, 0.6]])
        path = plot_roc_curve(y_true, y_prob, save_path=tmp_path / "roc.png")
        assert path.exists()

    def test_plot_multiclass(self, tmp_path: Path) -> None:
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_prob = np.array([
            [0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8],
            [0.6, 0.3, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7],
        ])
        path = plot_roc_curve(y_true, y_prob, save_path=tmp_path / "roc_multi.png")
        assert path.exists()


class TestPlotFeatureImportance:
    """Test feature importance plotting."""

    def test_plot(self, tmp_path: Path) -> None:
        importances = np.random.default_rng(42).random(20)
        names = [f"feat_{i}" for i in range(20)]
        path = plot_feature_importance(importances, feature_names=names, save_path=tmp_path / "fi.png")
        assert path.exists()


class TestCompareModels:
    """Test model comparison plotting."""

    def test_compare(self, tmp_path: Path) -> None:
        results = {
            "xgboost": {"accuracy": 0.85, "f1_macro": 0.82},
            "random_forest": {"accuracy": 0.80, "f1_macro": 0.78},
        }
        path = compare_models(results, save_path=tmp_path / "compare.png")
        assert path is not None
        assert path.exists()

    def test_compare_empty(self) -> None:
        result = compare_models({})
        assert result is None


# ======================================================================
# ModelTrainer
# ======================================================================

class TestModelTrainer:
    """Test ModelTrainer orchestration."""

    def test_train_classical(
        self, sample_features_2d: np.ndarray, sample_labels_2d: np.ndarray, tmp_path: Path,
    ) -> None:
        from models.trainer import ModelTrainer
        trainer = ModelTrainer(output_dir=tmp_path)
        result = trainer.train_classical(
            sample_features_2d, sample_labels_2d,
            model_type="random_forest", cv_folds=3,
        )
        assert "train_metrics" in result
        assert "eval_metrics" in result
        assert "xgboost" in trainer.results or "random_forest" in trainer.results

    def test_evaluate_model(
        self, sample_features_2d: np.ndarray, sample_labels_2d: np.ndarray, tmp_path: Path,
    ) -> None:
        from models.trainer import ModelTrainer
        trainer = ModelTrainer(output_dir=tmp_path)
        classifier = ClassicalClassifier(model_type="random_forest", n_classes=2)
        classifier.fit(sample_features_2d, sample_labels_2d, cv_folds=3)

        result = trainer.evaluate_model(classifier, sample_features_2d, sample_labels_2d)
        assert "metrics" in result
        assert "confusion_matrix_path" in result
        assert "roc_curve_path" in result
        assert "classification_report" in result

    def test_save_training_results(
        self, sample_features_2d: np.ndarray, sample_labels_2d: np.ndarray, tmp_path: Path,
    ) -> None:
        from models.trainer import ModelTrainer
        trainer = ModelTrainer(output_dir=tmp_path)
        trainer.train_classical(
            sample_features_2d, sample_labels_2d,
            model_type="random_forest", cv_folds=3,
        )
        result_path = trainer.save_training_results()
        assert result_path.exists()

        with open(result_path) as fh:
            data = json.load(fh)
        assert "random_forest" in data

    def test_hyperparameter_search(
        self, sample_features_2d: np.ndarray, sample_labels_2d: np.ndarray, tmp_path: Path,
    ) -> None:
        from models.trainer import ModelTrainer
        trainer = ModelTrainer(output_dir=tmp_path)
        param_grid = {
            "model__n_estimators": [50, 100],
            "model__max_depth": [3, 5],
        }
        result = trainer.hyperparameter_search(
            sample_features_2d, sample_labels_2d,
            param_grid=param_grid,
            model_type="random_forest",
            cv_folds=3,
        )
        assert "best_params" in result
        assert "best_score" in result
