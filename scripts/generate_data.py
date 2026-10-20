"""Data generation script for Voice Biomarker Platform v2.

Generates synthetic datasets with configurable size and class distribution.

Usage:
    python scripts/generate_data.py --num-samples 1000
    python scripts/generate_data.py --num-samples 2000 --unbalanced
    python scripts/generate_data.py --num-samples 500 --output-dir data/custom
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings

logger = logging.getLogger("voicebiomarker.generate_data")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic voice biomarker data")

    parser.add_argument(
        "--num-samples", type=int, default=1000,
        help="Total number of samples to generate (default: 1000)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: from config)",
    )
    parser.add_argument(
        "--target-sr", type=int, default=16000,
        help="Target sample rate (default: 16000)",
    )
    parser.add_argument(
        "--balanced", action="store_true", default=True,
        help="Generate balanced class distribution (default: True)",
    )
    parser.add_argument(
        "--unbalanced", action="store_true",
        help="Generate unbalanced class distribution",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify generated data after creation",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    return parser.parse_args()


def setup_logging(level: str) -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=getattr(logging, level), format=fmt, datefmt="%Y-%m-%d %H:%M:%S")


def verify_dataset(data_dir: Path) -> Dict[str, Any]:
    """Verify a generated dataset for consistency."""
    labels_path = data_dir / "labels.json"
    if not labels_path.exists():
        return {"valid": False, "error": "labels.json not found"}

    with open(labels_path) as fh:
        meta = json.load(fh)

    issues = []
    total_samples = meta.get("total_samples", 0)
    saved_samples = meta.get("samples", [])

    if len(saved_samples) != total_samples:
        issues.append(f"Label count mismatch: declared={total_samples}, actual={len(saved_samples)}")

    missing_files = 0
    for entry in saved_samples:
        audio_path = data_dir / entry["audio_file"]
        if not audio_path.exists():
            missing_files += 1

    if missing_files > 0:
        issues.append(f"Missing audio files: {missing_files}/{total_samples}")

    class_counts: Dict[str, int] = {}
    for entry in saved_samples:
        name = entry.get("label_name", "unknown")
        class_counts[name] = class_counts.get(name, 0) + 1

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "total_samples": total_samples,
        "saved_samples": len(saved_samples),
        "missing_files": missing_files,
        "class_distribution": class_counts,
        "target_sr": meta.get("target_sr", 0),
    }


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    settings = get_settings()
    output_dir = Path(args.output_dir or str(Path(settings.data.data_dir) / "synthetic"))
    balanced = not args.unbalanced

    logger.info(
        "Generating %d samples (balanced=%s, sr=%d, seed=%d)",
        args.num_samples, balanced, args.target_sr, args.seed,
    )

    from data.loaders import SyntheticLoader

    t0 = time.time()
    loader = SyntheticLoader(
        data_dir=output_dir,
        num_samples=args.num_samples,
        target_sr=args.target_sr,
        balanced=balanced,
        seed=args.seed,
    )
    samples = loader.load_dataset()
    elapsed = time.time() - t0

    metadata = loader.get_metadata()

    logger.info("Generated %d samples in %.1fs", len(samples), elapsed)

    # Print summary
    print("\n" + "=" * 60)
    print("DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total samples: {metadata['total_samples']}")
    print(f"Num classes:   {metadata['num_classes']}")
    print(f"Class names:   {metadata['class_names']}")
    print(f"Balanced:      {metadata['balanced']}")
    print(f"Target SR:     {metadata['target_sr']} Hz")
    print(f"Duration:      {elapsed:.1f}s")
    print(f"Output:        {output_dir}")
    print("\nClass distribution:")
    for name, count in metadata["class_counts"].items():
        pct = count / metadata["total_samples"] * 100
        print(f"  {name:20s}: {count:5d} ({pct:.1f}%)")

    if args.verify:
        print("\nVerifying dataset ...")
        verification = verify_dataset(output_dir)
        status = "PASS" if verification["valid"] else "FAIL"
        print(f"Verification: {status}")
        if verification["issues"]:
            for issue in verification["issues"]:
                print(f"  - {issue}")
        print(f"Audio files on disk: {verification['saved_samples'] - verification['missing_files']}/{verification['saved_samples']}")

    # Save splits info
    splits = loader.get_splits()
    print("\nSplit sizes:")
    for split_name, split_samples in splits.items():
        labels = [s["label_name"] for s in split_samples]
        dist = {}
        for l in labels:
            dist[l] = dist.get(l, 0) + 1
        print(f"  {split_name:5s}: {len(split_samples):5d} samples")
        for cls, cnt in dist.items():
            print(f"          {cls}: {cnt}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
