"""Model evaluation script for Voice Biomarker Platform v2.

Loads trained models and test data, runs comprehensive evaluation,
generates detailed reports, and compares with baseline models.

Usage:
    python scripts/evaluate.py --model-path models/checkpoints/xgboost_model.pkl --dataset synthetic
    python scripts/evaluate.py --model-path models/checkpoints/wav2vec2_model.pt --dataset synthetic
    python scripts/evaluate.py --compare --dataset synthetic
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.classical_models import ClassicalClassifier
from models.evaluation import (
    compute_metrics,
    compare_models,
    generate_classification_report,
    plot_confusion_matrix,
    plot_roc_curve,
)

logger = logging.getLogger("voicebiomarker.evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained voice biomarker models")

    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to trained model file",
    )
    parser.add_argument(
        "--dataset", type=str, default="synthetic",
        choices=["synthetic", "mpower", "daicwoz"],
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
    )
    parser.add_argument(
        "--output-dir", type=str, default="evaluation_results",
    )
    parser.add_argument(
        "--condition", type=str, default="parkinsons",
        choices=["parkinsons", "depression"],
    )
    parser.add_argument(
        "--num-samples", type=int, default=500,
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare all available models",
    )
    parser.add_argument(
        "--save-report", action="store_true",
        help="Save detailed evaluation report to JSON",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=getattr(logging, level), format=fmt, datefmt="%Y-%m-%d %H:%M:%S")
    for noisy in ("urllib3", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def load_test_data(
    dataset: str, data_dir: Optional[str], num_samples: int, seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load test split features and labels."""
    from config.settings import get_settings
    from data.loaders import SyntheticLoader
    from preprocessing.feature_extractor import FeatureExtractor

    settings = get_settings()
    actual_dir = data_dir or str(settings.data.data_dir)

    loader = SyntheticLoader(
        data_dir=actual_dir, num_samples=num_samples, seed=seed, balanced=True,
    )
    loader.load_dataset()
    splits = loader.get_splits(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

    extractor = FeatureExtractor()

    def _extract(samples: list) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for s in samples:
            features = extractor.extract_all(s["audio"], 16000)
            keys = sorted(k for k, v in features.items() if isinstance(v, (int, float)))
            X.append([float(features[k]) for k in keys])
            y.append(s["label"])
        return np.array(X), np.array(y)

    X_test, y_test = _extract(splits["test"])
    feature_names = sorted(
        k for k, v in extractor.extract_all(splits["train"][0]["audio"], 16000).items()
        if isinstance(v, (int, float))
    )
    return X_test, y_test, feature_names


def evaluate_single_model(
    model_path: str, X_test: np.ndarray, y_test: np.ndarray,
    feature_names: List[str], output_dir: Path,
) -> Dict[str, Any]:
    """Evaluate a single model and generate plots."""
    path = Path(model_path)

    if path.suffix == ".pkl":
        model = ClassicalClassifier.load(path)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        model_type = model._model_type
        try:
            importances = model.get_feature_importance()
        except (AttributeError, RuntimeError):
            importances = None
    elif path.suffix == ".pt":
        import torch
        from models.deep_models import Wav2Vec2Classifier
        model = Wav2Vec2Classifier.load(path)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            logits = model(X_t)
            y_prob = torch.softmax(logits, dim=1).cpu().numpy()
        y_pred = y_prob.argmax(axis=1)
        model_type = "wav2vec2"
        importances = None
    else:
        raise ValueError(f"Unsupported model format: {path.suffix}")

    metrics = compute_metrics(y_test, y_pred, y_prob)
    report_text = generate_classification_report(y_test, y_pred)

    output_dir.mkdir(parents=True, exist_ok=True)

    cm_path = plot_confusion_matrix(y_test, y_pred, save_path=output_dir / "confusion_matrix.png")
    roc_path = plot_roc_curve(y_test, y_prob, save_path=output_dir / "roc_curve.png")

    result = {
        "model_path": str(path),
        "model_type": model_type,
        "metrics": metrics,
        "classification_report": report_text,
        "confusion_matrix_path": str(cm_path),
        "roc_curve_path": str(roc_path),
        "num_test_samples": len(y_test),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    logger.info("Evaluation complete: accuracy=%.4f, f1=%.4f, roc_auc=%.4f",
                metrics["accuracy"], metrics["f1_macro"], metrics["roc_auc"])

    return result


def compare_all_baselines(
    X_test: np.ndarray, y_test: np.ndarray,
    feature_names: List[str], output_dir: Path,
) -> Dict[str, Any]:
    """Compare multiple classical model types as baselines."""
    from models.classical_models import SUPPORTED_MODEL_TYPES

    results = {}
    for model_type in SUPPORTED_MODEL_TYPES:
        logger.info("Evaluating baseline: %s", model_type)
        try:
            classifier = ClassicalClassifier(model_type=model_type, n_classes=len(np.unique(y_test)))
            classifier.fit(X_test, y_test, cv_folds=3)
            metrics = classifier.evaluate(X_test, y_test)
            results[model_type] = metrics
            logger.info("  %s -> accuracy=%.4f", model_type, metrics["accuracy"])
        except Exception as exc:
            logger.warning("  %s failed: %s", model_type, exc)

    if results:
        compare_path = output_dir / "model_comparison.png"
        compare_models(results, save_path=compare_path)
        logger.info("Comparison chart saved to %s", compare_path)

    return results


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    logger.info("Loading test data from %s ...", args.dataset)
    X_test, y_test, feature_names = load_test_data(
        args.dataset, args.data_dir, args.num_samples, args.seed,
    )
    logger.info("Test set: %d samples, %d features", len(X_test), X_test.shape[1])

    all_results = {}

    if args.model_path:
        t0 = time.time()
        result = evaluate_single_model(
            args.model_path, X_test, y_test, feature_names, output_dir,
        )
        all_results["model"] = result
        logger.info("Evaluation took %.1fs", time.time() - t0)

    if args.compare:
        logger.info("Running baseline comparison ...")
        baseline_results = compare_all_baselines(X_test, y_test, feature_names, output_dir)
        all_results["baselines"] = baseline_results

    if args.save_report:
        report_path = output_dir / "evaluation_report.json"
        with open(report_path, "w") as fh:
            json.dump(all_results, fh, indent=2, default=str)
        logger.info("Full report saved to %s", report_path)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    if "model" in all_results:
        m = all_results["model"]["metrics"]
        print(f"Model: {all_results['model']['model_type']}")
        print(f"  Accuracy:     {m['accuracy']:.4f}")
        print(f"  F1 Macro:     {m['f1_macro']:.4f}")
        print(f"  Precision:    {m['precision_macro']:.4f}")
        print(f"  Recall:       {m['recall_macro']:.4f}")
        print(f"  ROC-AUC:      {m['roc_auc']:.4f}")
    if "baselines" in all_results:
        print("\nBaseline comparison:")
        for name, m in all_results["baselines"].items():
            print(f"  {name:25s} acc={m['accuracy']:.4f}  f1={m['f1_macro']:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
