"""Main training script for Voice Biomarker Platform v2.

Supports classical (xgboost/svm/rf) and deep (wav2vec2) models with
cross-validation, hyperparameter tuning, and evaluation plot generation.

Usage:
    python scripts/train.py --model-type xgboost --dataset synthetic --epochs 100
    python scripts/train.py --model-type wav2vec2 --dataset synthetic --epochs 50
    python scripts/train.py --model-type xgboost --tune --cv-folds 5
    python scripts/train.py --resume models/checkpoints/checkpoint.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import TrainingConfig, get_settings
from models.classical_models import ClassicalClassifier, SUPPORTED_MODEL_TYPES
from models.evaluation import (
    compute_metrics,
    generate_classification_report,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)
from models.trainer import ModelTrainer

logger = logging.getLogger("voicebiomarker.train")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train voice biomarker classification models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--model-type",
        type=str,
        default="xgboost",
        choices=["xgboost", "svm", "random_forest", "gradient_boosting", "wav2vec2"],
        help="Model type to train (default: xgboost)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        choices=["synthetic", "mpower", "daicwoz"],
        help="Dataset to use (default: synthetic)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/checkpoints",
        help="Directory to save trained models (default: models/checkpoints)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="parkinsons",
        choices=["parkinsons", "depression"],
        help="Condition to train for (default: parkinsons)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of synthetic samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs (overrides config, for deep models)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable hyperparameter tuning with GridSearchCV",
    )
    parser.add_argument(
        "--param-grid",
        type=str,
        default=None,
        help="JSON file with hyperparameter grid for tuning",
    )
    parser.add_argument(
        "--search-type",
        type=str,
        default="grid",
        choices=["grid", "random"],
        help="Search type for hyperparameter tuning (default: grid)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint JSON to resume training from",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating evaluation plots",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=getattr(logging, level), format=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    for noisy in ("urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_synthetic_data(
    data_dir: str,
    num_samples: int,
    seed: int,
    condition: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load synthetic data and return feature matrices and labels."""
    from data.loaders import SyntheticLoader

    loader = SyntheticLoader(
        data_dir=data_dir,
        num_samples=num_samples,
        seed=seed,
        balanced=True,
    )
    loader.load_dataset()
    splits = loader.get_splits(
        train_ratio=1.0 - 0.15 - 0.15,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    from preprocessing.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor()

    def _extract_features(samples: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for s in samples:
            audio = s["audio"]
            features = extractor.extract_all(audio, 16000)
            scalar_keys = sorted(k for k, v in features.items() if isinstance(v, (int, float)))
            X_list.append([float(features[k]) for k in scalar_keys])
            y_list.append(s["label"])
        return np.array(X_list), np.array(y_list)

    X_train, y_train = _extract_features(splits["train"])
    X_val, y_val = _extract_features(splits["val"])
    X_test, y_test = _extract_features(splits["test"])

    feature_names = sorted(
        k for k, v in extractor.extract_all(splits["train"][0]["audio"], 16000).items()
        if isinstance(v, (int, float))
    )

    logger.info(
        "Loaded synthetic data: train=%d, val=%d, test=%d, features=%d",
        len(X_train), len(X_val), len(X_test), X_train.shape[1],
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


def load_real_data(
    dataset: str,
    data_dir: str,
    seed: int,
    condition: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Load real dataset (mpower or daicwoz)."""
    if dataset == "mpower":
        from data.loaders import MPowerLoader
        loader = MPowerLoader(data_dir)
    elif dataset == "daicwoz":
        from data.loaders import DAICWOZLoader
        loader = DAICWOZLoader(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    loader.load_dataset()
    splits = loader.get_splits()

    from preprocessing.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor()

    def _extract_features(samples: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for s in samples:
            audio = s["audio"]
            features = extractor.extract_all(audio, 16000)
            scalar_keys = sorted(k for k, v in features.items() if isinstance(v, (int, float)))
            X_list.append([float(features[k]) for k in scalar_keys])
            y_list.append(s["label"])
        return np.array(X_list), np.array(y_list)

    X_train, y_train = _extract_features(splits["train"])
    X_val, y_val = _extract_features(splits["val"])
    X_test, y_test = _extract_features(splits["test"])

    feature_names = sorted(
        k for k, v in extractor.extract_all(splits["train"][0]["audio"], 16000).items()
        if isinstance(v, (int, float))
    )

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_names


# ---------------------------------------------------------------------------
# Classical training
# ---------------------------------------------------------------------------

def train_classical(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    output_dir: Path,
    cv_folds: int,
    do_tune: bool,
    param_grid: Optional[Dict[str, List[Any]]],
    search_type: str,
    n_classes: int,
    seed: int,
) -> Dict[str, Any]:
    """Train a classical model, optionally tune, and evaluate."""
    trainer = ModelTrainer(output_dir=output_dir)

    # Hyperparameter tuning
    if do_tune:
        grid = param_grid or _default_param_grid(model_type)
        logger.info("Running %s hyperparameter search ...", search_type)
        search_result = trainer.hyperparameter_search(
            X_train, y_train,
            param_grid=grid,
            model_type=model_type,
            search_type=search_type,
            cv_folds=cv_folds,
        )
        logger.info("Best params: %s (score=%.4f)", search_result["best_params"], search_result["best_score"])

    # Train
    logger.info("Training %s classifier ...", model_type)
    result = trainer.train_classical(
        X_train, y_train,
        model_type=model_type,
        cv_folds=cv_folds,
    )

    # Evaluate on test set
    classifier = ClassicalClassifier(model_type=model_type, n_classes=n_classes)
    classifier.fit(X_train, y_train, cv_folds=cv_folds)
    test_metrics = classifier.evaluate(X_test, y_test)
    y_pred = classifier.predict(X_test)
    y_prob = classifier.predict_proba(X_test)

    logger.info("Test metrics: %s", json.dumps(test_metrics, indent=2))

    # Save model
    model_path = output_dir / f"{model_type}_model.pkl"
    classifier.save(model_path)
    logger.info("Model saved to %s", model_path)

    # Generate plots
    cm_path = plot_confusion_matrix(
        y_test, y_pred,
        class_names=sorted(set(y_test.astype(str))),
        save_path=output_dir / "confusion_matrix.png",
    )
    roc_path = plot_roc_curve(
        y_test, y_prob,
        save_path=output_dir / "roc_curve.png",
    )

    feature_imp_path = None
    try:
        importances = classifier.get_feature_importance()
        feature_imp_path = plot_feature_importance(
            importances, feature_names=feature_names, top_n=20,
            save_path=output_dir / "feature_importance.png",
        )
    except (AttributeError, RuntimeError) as exc:
        logger.warning("Could not plot feature importance: %s", exc)

    # Build full report
    report = {
        "model_type": model_type,
        "train_metrics": result,
        "test_metrics": test_metrics,
        "classification_report": generate_classification_report(y_test, y_pred),
        "confusion_matrix_path": str(cm_path),
        "roc_curve_path": str(roc_path),
        "feature_importance_path": str(feature_imp_path) if feature_imp_path else None,
        "model_path": str(model_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    trainer.results["test"] = report
    trainer.results["test"]["metrics"] = test_metrics
    trainer.save_training_results()

    return report


def _default_param_grid(model_type: str) -> Dict[str, List[Any]]:
    """Return a sensible default parameter grid per model type."""
    grids = {
        "xgboost": {
            "model__n_estimators": [100, 200, 300],
            "model__max_depth": [3, 5, 7],
            "model__learning_rate": [0.01, 0.1, 0.2],
            "model__subsample": [0.8, 1.0],
        },
        "svm": {
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["rbf", "linear"],
            "model__gamma": ["scale", "auto"],
        },
        "random_forest": {
            "model__n_estimators": [100, 200, 500],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_split": [2, 5, 10],
        },
        "gradient_boosting": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [3, 5],
            "model__learning_rate": [0.01, 0.1],
        },
    }
    return grids.get(model_type, {})


# ---------------------------------------------------------------------------
# Deep model training
# ---------------------------------------------------------------------------

def train_deep(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    n_classes: int,
    patience: int,
) -> Dict[str, Any]:
    """Train a Wav2Vec2 deep classifier."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from models.deep_models import Wav2Vec2Classifier
    from models.trainer import ModelTrainer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training Wav2Vec2 classifier on %s", device)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False,
    )

    model = Wav2Vec2Classifier(
        num_classes=n_classes,
        freeze_base=True,
        dropout=0.3,
    )

    trainer = ModelTrainer(output_dir=output_dir)
    history = trainer.train_deep(
        model, train_loader, val_loader,
        epochs=epochs,
        lr=learning_rate,
        patience=patience,
        device=device,
    )

    test_metrics = model.evaluate(test_loader, device=device)
    logger.info("Test metrics: %s", json.dumps(test_metrics, indent=2))

    model_path = output_dir / "wav2vec2_model.pt"
    model.save(model_path)
    logger.info("Model saved to %s", model_path)

    trainer.plot_training_history(history)

    report = {
        "model_type": "wav2vec2",
        "train_history": history,
        "test_metrics": test_metrics,
        "model_path": str(model_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    trainer.results["deep_test"] = report
    trainer.save_training_results()

    return report


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------

def save_checkpoint(output_dir: Path, args: argparse.Namespace, epoch: int) -> None:
    """Save a training checkpoint for resuming."""
    ckpt = {
        "args": vars(args),
        "epoch": epoch,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    ckpt_path = output_dir / "checkpoint.json"
    with open(ckpt_path, "w") as fh:
        json.dump(ckpt, fh, indent=2)
    logger.info("Checkpoint saved to %s", ckpt_path)


def load_checkpoint(resume_path: str) -> Dict[str, Any]:
    """Load a training checkpoint."""
    with open(resume_path) as fh:
        ckpt = json.load(fh)
    logger.info("Resumed checkpoint from epoch %d", ckpt.get("epoch", 0))
    return ckpt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    settings = get_settings()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = load_checkpoint(args.resume)
        start_epoch = ckpt.get("epoch", 0)
        logger.info("Resuming from epoch %d", start_epoch)

    logger.info(
        "Training %s model on %s dataset (condition=%s, cv=%d)",
        args.model_type, args.dataset, args.condition, args.cv_folds,
    )

    # Load data
    t0 = time.time()
    if args.dataset == "synthetic":
        data_dir = args.data_dir or str(settings.data.data_dir)
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_synthetic_data(
            data_dir=data_dir,
            num_samples=args.num_samples,
            seed=args.seed,
            condition=args.condition,
        )
    else:
        data_dir = args.data_dir or str(Path(settings.data.data_dir) / args.dataset)
        X_train, y_train, X_val, y_val, X_test, y_test, feature_names = load_real_data(
            dataset=args.dataset,
            data_dir=data_dir,
            seed=args.seed,
            condition=args.condition,
        )

    n_classes = len(np.unique(y_train))
    data_time = time.time() - t0
    logger.info("Data loaded in %.1fs", data_time)

    # Train
    t1 = time.time()
    if args.model_type == "wav2vec2":
        report = train_deep(
            X_train, y_train, X_val, y_val, X_test, y_test,
            output_dir=output_dir,
            epochs=args.epochs or settings.training.epochs,
            batch_size=args.batch_size or settings.training.batch_size,
            learning_rate=args.learning_rate or settings.training.learning_rate,
            n_classes=n_classes,
            patience=settings.training.early_stopping_patience,
        )
    else:
        report = train_classical(
            model_type=args.model_type,
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            feature_names=feature_names,
            output_dir=output_dir,
            cv_folds=args.cv_folds,
            do_tune=args.tune,
            param_grid=json.loads(args.param_grid) if args.param_grid else None,
            search_type=args.search_type,
            n_classes=n_classes,
            seed=args.seed,
        )

    train_time = time.time() - t1

    # Save final checkpoint
    save_checkpoint(output_dir, args, start_epoch + 1)

    logger.info("Training complete in %.1fs", train_time)
    logger.info("Results saved to %s", output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Model: {args.model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Condition: {args.condition}")
    if "test_metrics" in report:
        print(f"Test Accuracy: {report['test_metrics']['accuracy']:.4f}")
        print(f"Test F1 Macro: {report['test_metrics']['f1_macro']:.4f}")
        print(f"Test ROC-AUC: {report['test_metrics']['roc_auc']:.4f}")
    elif "test_metrics" in report:
        for k, v in report["test_metrics"].items():
            print(f"  {k}: {v}")
    print(f"Output: {output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
