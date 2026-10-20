"""Loader for the DAIC-WOZ Depression dataset.

Handles the DAIC-WOZ dataset structure (interview audio, PHQ-8 depression
scores, train/dev/test splits as provided by the dataset, and audio/video
file formats).

Data-Use Agreement Notes
------------------------
The DAIC-WOZ dataset is part of the NIMH Archive (NDA) and distributed
through the University of Southern California's Institute for Creative
Technologies.  Before using this loader:

1. Request access at https://nda.nih.gov/ndar/project/details/1686
2. Accept the Data Use Agreement (DUA).
3. Download data manually (the dataset ships with ``train_split.csv``,
   ``dev_split.csv``, and ``test_split.csv``).

This loader does **not** auto-download the dataset.

Expected Directory Structure
----------------------------
::

    data/daicwoz/
    ├── train_split.csv
    ├── dev_split.csv
    ├── test_split.csv
    ├── Full_wave_audio/
    │   ├── <participant_id>_P/
    │   │   ├── <participant_id>_P.wav
    │   │   └── ...
    │   └── ...
    └── Full_wave_aligned/
        └── ...  (optional aligned audio)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from data.loaders.base_loader import BaseLoader
from data.loaders.utils import LabelEncoder, save_json

logger = logging.getLogger(__name__)

# PHQ-8 depression severity thresholds (consistent with clinical cutoffs)
PHQ8_THRESHOLDS: Dict[str, float] = {
    "minimal": 0,       # 0–4
    "mild": 5,          # 5–9
    "moderate": 10,     # 10–14
    "moderately_severe": 15,  # 15–19
    "severe": 20,       # 20–24
}


def phq8_label(score: float) -> str:
    """Map a PHQ-8 score to a severity label."""
    if score >= PHQ8_THRESHOLDS["severe"]:
        return "depression_severe"
    if score >= PHQ8_THRESHOLDS["moderately_severe"]:
        return "depression_moderately_severe"
    if score >= PHQ8_THRESHOLDS["moderate"]:
        return "depression_moderate"
    if score >= PHQ8_THRESHOLDS["mild"]:
        return "depression_mild"
    return "no_depression"


def phq8_binary(score: float, threshold: float = 10.0) -> str:
    """Binary depression label: ``"depressed"`` vs ``"not_depressed"``."""
    return "depressed" if score >= threshold else "not_depressed"


class DAICWOZLoader(BaseLoader):
    """Loader for the DAIC-WOZ depression interview dataset.

    Parameters
    ----------
    data_dir:
        Root directory of the DAIC-WOZ dataset.
    label_mode:
        ``"binary"`` for depressed/not-depressed, ``"multiclass"`` for
        PHQ-8 severity bins, or ``"regression"`` to return raw PHQ-8 scores.
    phq8_threshold:
        Threshold for binary labelling (only used when
        ``label_mode="binary"``).
    target_sr:
        Target sample rate for audio loading.
    max_duration:
        Maximum audio duration in seconds.
    seed:
        Random seed for reproducible splits.

    Examples
    --------
    >>> loader = DAICWOZLoader("data/daicwoz")
    >>> loader.load_dataset()
    >>> meta = loader.get_metadata()
    >>> print(meta["phq8_statistics"])
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/daicwoz",
        *,
        label_mode: str = "binary",
        phq8_threshold: float = 10.0,
        target_sr: int = 16000,
        max_duration: int = 30,
        seed: int = 42,
    ) -> None:
        super().__init__(
            data_dir,
            target_sr=target_sr,
            max_duration=max_duration,
            seed=seed,
        )
        if label_mode not in {"binary", "multiclass", "regression"}:
            raise ValueError(
                f"Invalid label_mode '{label_mode}'. "
                "Choose from 'binary', 'multiclass', or 'regression'."
            )
        self.label_mode = label_mode
        self.phq8_threshold = phq8_threshold

    # ------------------------------------------------------------------
    # BaseLoader interface
    # ------------------------------------------------------------------

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the DAIC-WOZ dataset across all available splits.

        Iterates over ``train_split.csv``, ``dev_split.csv``, and
        ``test_split.csv`` (whichever exist) to build the full dataset.
        """
        split_files = {
            "train": self.data_dir / "train_split.csv",
            "dev": self.data_dir / "dev_split.csv",
            "test": self.data_dir / "test_split.csv",
        }

        all_rows: List[Dict[str, str]] = []
        self._split_map: Dict[int, str] = {}  # sample_id -> split

        for split_name, csv_path in split_files.items():
            if not csv_path.exists():
                logger.warning("Split file not found, skipping: %s", csv_path)
                continue
            rows = self._read_split_csv(csv_path)
            for row in rows:
                sample_id = len(all_rows)
                row["_split"] = split_name
                row["_sample_id"] = sample_id
                self._split_map[sample_id] = split_name
                all_rows.append(row)

        if not all_rows:
            raise FileNotFoundError(
                f"No split CSV files found in {self.data_dir}. "
                "Please download the DAIC-WOZ dataset first."
            )

        logger.info(
            "Parsed %d participants across splits: %s",
            len(all_rows),
            {s: sum(1 for r in all_rows if r["_split"] == s) for s in split_files},
        )

        # Assign labels
        all_rows = [r for r in all_rows if self._has_valid_phq8(r)]

        # Fit label encoder
        if self.label_mode != "regression":
            label_names = [self._score_to_label(r) for r in all_rows]
            self._label_encoder.fit(list(set(label_names)))

        # Load audio
        samples: List[Dict[str, Any]] = []
        skipped = 0
        for row in all_rows:
            audio_path = self._resolve_audio_path(row)
            if audio_path is None or not audio_path.exists():
                skipped += 1
                continue

            try:
                audio = self.load_audio(audio_path)
            except Exception as exc:
                logger.debug("Skipping participant %s: %s", row.get("participant_id", "?"), exc)
                skipped += 1
                continue

            phq8_score = self._get_phq8_score(row)
            label_name = self._score_to_label(row)
            label_idx = (
                self._label_encoder.encode(label_name)
                if self.label_mode != "regression"
                else -1  # placeholder for regression
            )

            samples.append(
                {
                    "id": row["_sample_id"],
                    "audio": audio,
                    "label": label_idx,
                    "label_name": label_name,
                    "duration_sec": len(audio) / self.target_sr,
                    "source": "daicwoz",
                    "split": row["_split"],
                    "metadata": {
                        "participant_id": row.get("participant_id", ""),
                        "phq8_score": phq8_score,
                        "gender": row.get("gender", ""),
                        "age": row.get("age", ""),
                        "split": row["_split"],
                    },
                }
            )

        if skipped:
            logger.warning(
                "Skipped %d participants (missing audio or load errors)", skipped
            )

        logger.info("Loaded %d valid samples.", len(samples))
        return samples

    def get_splits(
        self,
        *,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Return train/val/test splits.

        DAIC-WOZ provides its own ``train``/``dev``/``test`` splits.
        This loader respects those splits.  The ``dev`` split is mapped
        to ``val``.  The ``train_ratio``, ``val_ratio``, and
        ``test_ratio`` parameters are ignored unless the split
        information is missing.
        """
        samples = self._ensure_loaded()

        # Check if split info is available
        has_splits = any("split" in s for s in samples)

        if has_splits:
            result: Dict[str, List[Dict[str, Any]]] = {
                "train": [], "val": [], "test": [],
            }
            for s in samples:
                sp = s.get("split", "train")
                if sp in ("dev", "validation"):
                    sp = "val"
                elif sp not in ("train", "val", "test"):
                    sp = "train"
                result[sp].append(s)
            return result
        else:
            # Fallback: stratified split
            return self._stratified_split(
                samples,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )

    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset metadata and PHQ-8 statistics."""
        samples = self._ensure_loaded()

        phq8_scores = [
            s["metadata"]["phq8_score"]
            for s in samples
            if s["metadata"].get("phq8_score") is not None
        ]
        class_counts: Dict[str, int] = {}
        split_counts: Dict[str, int] = {}
        genders: List[str] = []

        for s in samples:
            name = s["label_name"]
            class_counts[name] = class_counts.get(name, 0) + 1
            sp = s.get("split", "unknown")
            split_counts[sp] = split_counts.get(sp, 0) + 1
            g = s["metadata"].get("gender", "")
            if g:
                genders.append(g)

        phq8_arr = np.array(phq8_scores) if phq8_scores else np.array([0.0])

        return {
            "dataset": "daicwoz_depression",
            "label_mode": self.label_mode,
            "phq8_threshold": self.phq8_threshold,
            "total_samples": len(samples),
            "num_classes": self.num_classes if self.label_mode != "regression" else 0,
            "class_names": self.class_names if self.label_mode != "regression" else [],
            "class_counts": class_counts,
            "split_counts": split_counts,
            "phq8_statistics": {
                "mean": float(np.mean(phq8_arr)),
                "std": float(np.std(phq8_arr)),
                "min": float(np.min(phq8_arr)),
                "max": float(np.max(phq8_arr)),
                "median": float(np.median(phq8_arr)),
                "count": len(phq8_scores),
            },
            "gender_distribution": {g: genders.count(g) for g in set(genders)} if genders else {},
        }

    # ------------------------------------------------------------------
    # CSV parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _read_split_csv(csv_path: Path) -> List[Dict[str, str]]:
        """Read a DAIC-WOZ split CSV."""
        rows: List[Dict[str, str]] = []
        with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(dict(row))
        return rows

    def _has_valid_phq8(self, row: Dict[str, str]) -> bool:
        """Check if a row has a valid PHQ-8 score."""
        score = self._get_phq8_score(row)
        return score is not None and 0 <= score <= 24

    def _get_phq8_score(self, row: Dict[str, str]) -> Optional[float]:
        """Extract the PHQ-8 score from a CSV row."""
        # Try various column name conventions
        for col in ("PHQ8_Score", "phq8_score", "PHQ-8", "phq8", "PHQ8Total"):
            val = row.get(col, "").strip()
            if val:
                try:
                    return float(val)
                except (ValueError, TypeError):
                    continue
        return None

    def _score_to_label(self, row: Dict[str, str]) -> str:
        """Map a CSV row to a label name."""
        score = self._get_phq8_score(row)
        if score is None:
            return "unknown"
        if self.label_mode == "binary":
            return phq8_binary(score, self.phq8_threshold)
        elif self.label_mode == "multiclass":
            return phq8_label(score)
        else:
            return f"phq8_{score}"

    def _resolve_audio_path(self, row: Dict[str, str]) -> Optional[Path]:
        """Try to find the audio file for a given participant."""
        pid = row.get("participant_id", "").strip()
        if not pid:
            # Try other ID fields
            pid = row.get("participant", "").strip()
            if not pid:
                pid = row.get("id", "").strip()

        if not pid:
            return None

        # Ensure PID format is consistent (DAIC-WOZ uses e.g. "300_P")
        if not pid.endswith("_P"):
            pid_with_suffix = f"{pid}_P"
        else:
            pid_with_suffix = pid

        # Strategy 1: Full_wave_audio/<pid>/<pid>.wav
        for audio_dir in ("Full_wave_audio", "Full_wave_aligned", "audio"):
            candidate = self.data_dir / audio_dir / pid_with_suffix / f"{pid_with_suffix}.wav"
            if candidate.exists():
                return candidate

            # Also check with just the PID as filename
            candidate2 = self.data_dir / audio_dir / pid_with_suffix / f"{pid}.wav"
            if candidate2.exists():
                return candidate2

        # Strategy 2: Search for any .wav file matching the PID
        for fpath in self.data_dir.rglob(f"*{pid}*.wav"):
            if fpath.is_file():
                return fpath

        # Strategy 3: Check for .mp3 or .mp4 audio extraction
        for ext in (".mp3", ".mp4", ".m4a", ".flac"):
            for fpath in self.data_dir.rglob(f"*{pid}*{ext}"):
                if fpath.is_file():
                    return fpath

        logger.debug("No audio file found for participant %s", pid)
        return None

    def get_phq8_scores(self) -> Dict[int, float]:
        """Return a mapping of sample ID to PHQ-8 score."""
        samples = self._ensure_loaded()
        return {
            s["id"]: s["metadata"]["phq8_score"]
            for s in samples
            if s["metadata"].get("phq8_score") is not None
        }
