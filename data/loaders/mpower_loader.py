"""Loader for the mPower Parkinson's dataset.

Handles the mPower study dataset structure (CSV metadata + audio recordings),
extracts relevant clinical columns, handles missing data, and provides
age/gender metadata.

Data-Use Agreement Notes
------------------------
The mPower dataset is distributed through Sage Bionetworks under a
**Data Use Limitation (DUL)** agreement.  Before using this loader:

1. Register at https://www.synapse.org/#!Synapse:syn4993293
2. Accept the Data Use Certificate (DUC).
3. Download data manually or via ``synapse get`` CLI.

This loader does **not** auto-download the dataset.  It expects the
data to already be present in the directory structure described below.

Expected Directory Structure
----------------------------
::

    data/mpower/
    ├── mPowerRelease1/
    │   ├── mPower.csv                  # Main metadata CSV
    │   ├── audio_video_responses/      # Audio/video recordings
    │   │   ├── <participant_id>/
    │   │   │   ├── <task>_*.wav
    │   │   │   └── ...
    │   │   └── ...
    │   └── demographics.csv            # Optional demographics
    └── README.txt                      # Dataset documentation
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from data.loaders.base_loader import BaseLoader
from data.loaders.utils import LabelEncoder, save_json

logger = logging.getLogger(__name__)

# Columns we care about from the main CSV
_DESIRED_COLUMNS = {
    "participant_id",
    "record_id",
    "updrs_score",
    "recording",
    "task",
    "age",
    "gender",
    "handedness",
    "professional_diagnosis",
}


class MPowerLoader(BaseLoader):
    """Loader for the mPower Parkinson's disease dataset.

    Parameters
    ----------
    data_dir:
        Root directory of the mPower dataset.
    task:
        Which mPower task to load (``"voice"``, ``"tapping"``,
        ``"balance"``, ``"gait"``).  Defaults to ``"voice"``.
    updrs_threshold:
        UPDRS score above which a participant is considered "affected".
        Participants below are labelled ``"healthy"``.
    target_sr:
        Target sample rate for audio loading.
    max_duration:
        Maximum audio duration in seconds.
    seed:
        Random seed for reproducible splits.
    download_hint:
        If *True*, print download instructions when data is missing.

    Examples
    --------
    >>> loader = MPowerLoader("data/mpower")
    >>> loader.load_dataset()
    >>> meta = loader.get_metadata()
    >>> print(meta["class_counts"])
    """

    def __init__(
        self,
        data_dir: Union[str, Path] = "data/mpower",
        *,
        task: str = "voice",
        updrs_threshold: float = 20.0,
        target_sr: int = 16000,
        max_duration: int = 30,
        seed: int = 42,
        download_hint: bool = True,
    ) -> None:
        super().__init__(
            data_dir,
            target_sr=target_sr,
            max_duration=max_duration,
            seed=seed,
        )
        self.task = task
        self.updrs_threshold = updrs_threshold
        self.download_hint = download_hint

    # ------------------------------------------------------------------
    # BaseLoader interface
    # ------------------------------------------------------------------

    def load_dataset(self) -> List[Dict[str, Any]]:
        """Load the mPower dataset.

        Reads the CSV metadata, maps recordings to labels based on the
        UPDRS score, and loads the audio files.
        """
        csv_path = self._find_metadata_csv()
        if csv_path is None:
            if self.download_hint:
                self._print_download_instructions()
            raise FileNotFoundError(
                f"mPower metadata CSV not found in {self.data_dir}. "
                "See download instructions above."
            )

        rows = self._read_csv(csv_path)
        logger.info("Read %d rows from %s", len(rows), csv_path)

        # Filter to the requested task
        rows = self._filter_task(rows)
        logger.info("After task filter (%s): %d rows", self.task, len(rows))

        # Assign labels
        rows = self._assign_labels(rows)

        # Build label encoder
        all_labels = [r["_label_name"] for r in rows]
        self._label_encoder.fit(list(set(all_labels)))

        # Load audio
        samples: List[Dict[str, Any]] = []
        skipped = 0
        for i, row in enumerate(rows):
            audio_path = self._resolve_audio_path(row)
            if audio_path is None or not audio_path.exists():
                skipped += 1
                continue

            try:
                audio = self.load_audio(audio_path)
            except Exception as exc:
                logger.debug("Skipping sample %d: %s", i, exc)
                skipped += 1
                continue

            label_idx = self._label_encoder.encode(row["_label_name"])
            samples.append(
                {
                    "id": i,
                    "audio": audio,
                    "label": label_idx,
                    "label_name": row["_label_name"],
                    "duration_sec": len(audio) / self.target_sr,
                    "source": "mpower",
                    "metadata": {
                        "participant_id": row.get("participant_id", ""),
                        "updrs_score": row.get("updrs_score"),
                        "age": row.get("age"),
                        "gender": row.get("gender"),
                        "task": row.get("task", self.task),
                    },
                }
            )

        if skipped:
            logger.warning(
                "Skipped %d samples (missing files or load errors)", skipped
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
        """Return stratified train/val/test splits."""
        samples = self._ensure_loaded()
        return self._stratified_split(
            samples,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

    def get_metadata(self) -> Dict[str, Any]:
        """Return dataset metadata and demographics summary."""
        samples = self._ensure_loaded()

        ages: List[float] = []
        genders: List[str] = []
        class_counts: Dict[str, int] = {}
        updrs_scores: List[float] = []

        for s in samples:
            name = s["label_name"]
            class_counts[name] = class_counts.get(name, 0) + 1
            md = s.get("metadata", {})
            if md.get("age") is not None:
                try:
                    ages.append(float(md["age"]))
                except (ValueError, TypeError):
                    pass
            if md.get("gender"):
                genders.append(md["gender"])
            if md.get("updrs_score") is not None:
                try:
                    updrs_scores.append(float(md["updrs_score"]))
                except (ValueError, TypeError):
                    pass

        return {
            "dataset": "mpower_parkinson",
            "total_samples": len(samples),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "class_counts": class_counts,
            "task": self.task,
            "updrs_threshold": self.updrs_threshold,
            "age_mean": float(np.mean(ages)) if ages else None,
            "age_std": float(np.std(ages)) if ages else None,
            "age_range": (float(min(ages)), float(max(ages))) if ages else None,
            "gender_distribution": {
                g: genders.count(g) for g in set(genders)
            }
            if genders
            else {},
            "updrs_mean": float(np.mean(updrs_scores)) if updrs_scores else None,
            "updrs_std": float(np.std(updrs_scores)) if updrs_scores else None,
        }

    # ------------------------------------------------------------------
    # CSV parsing
    # ------------------------------------------------------------------

    def _find_metadata_csv(self) -> Optional[Path]:
        """Search for the main metadata CSV."""
        candidates = [
            self.data_dir / "mPowerRelease1" / "mPower.csv",
            self.data_dir / "mPower.csv",
            self.data_dir / "data.csv",
        ]
        for p in candidates:
            if p.exists():
                return p

        # Fallback: find any CSV with "mPower" or "updrs" in the name
        for p in self.data_dir.rglob("*.csv"):
            name_lower = p.name.lower()
            if "mpower" in name_lower or "updrs" in name_lower:
                return p

        return None

    @staticmethod
    def _read_csv(csv_path: Path) -> List[Dict[str, str]]:
        """Read a CSV file into a list of dicts."""
        rows: List[Dict[str, str]] = []
        with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(dict(row))
        return rows

    def _filter_task(self, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Keep only rows matching the requested task."""
        task_field = None
        # Discover which column holds the task name
        if rows and "task" in rows[0]:
            task_field = "task"
        elif rows and "Activity" in rows[0]:
            task_field = "Activity"

        if task_field is None:
            logger.warning("No task column found; returning all rows.")
            return rows

        return [r for r in rows if self.task.lower() in r.get(task_field, "").lower()]

    def _assign_labels(self, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Assign ``_label_name`` based on the UPDRS score.

        - UPDRS >= threshold  -->  ``"parkinson"``
        - UPDRS <  threshold  -->  ``"healthy"``
        - UPDRS missing       -->  ``"unknown"``
        """
        for row in rows:
            updrs_raw = row.get("updrs_score", "").strip()
            try:
                updrs = float(updrs_raw)
            except (ValueError, TypeError):
                row["_label_name"] = "unknown"
                continue

            row["_label_name"] = "parkinson" if updrs >= self.updrs_threshold else "healthy"

        # Remove unknowns
        return [r for r in rows if r["_label_name"] != "unknown"]

    def _resolve_audio_path(self, row: Dict[str, str]) -> Optional[Path]:
        """Try to find the audio file for a given CSV row."""
        rec = row.get("recording", "").strip()
        participant = row.get("participant_id", "").strip()
        record_id = row.get("record_id", "").strip()

        # Strategy 1: direct path in the recording column
        if rec:
            candidate = self.data_dir / "mPowerRelease1" / "audio_video_responses" / rec
            if candidate.exists():
                return candidate

        # Strategy 2: participant_id / record_id pattern
        if participant and record_id:
            audio_dir = self.data_dir / "mPowerRelease1" / "audio_video_responses" / participant
            if audio_dir.is_dir():
                for fpath in audio_dir.iterdir():
                    if fpath.is_file() and fpath.suffix.lower() in {".wav", ".mp3", ".ogg"}:
                        if record_id in fpath.stem:
                            return fpath
                    # Fallback: first audio file in the directory
                for fpath in audio_dir.iterdir():
                    if fpath.is_file() and fpath.suffix.lower() in {".wav", ".mp3", ".ogg"}:
                        return fpath

        # Strategy 3: glob search by participant
        if participant:
            for fpath in self.data_dir.rglob(f"*{participant}*"):
                if fpath.is_file() and fpath.suffix.lower() in {".wav", ".mp3", ".ogg"}:
                    return fpath

        return None

    # ------------------------------------------------------------------
    # Download instructions
    # ------------------------------------------------------------------

    def _print_download_instructions(self) -> None:
        instructions = """
=====================================================================
  mPower Parkinson's Dataset – Download Instructions
=====================================================================

1. Visit https://www.synapse.org/#!Synapse:syn4993293
2. Create a free Synapse account (or log in).
3. Accept the Data Use Certificate (DUC) for Study syn4993293.
4. Download the dataset:
   a) Via the web UI: click "Download" on the main page.
   b) Via the CLI:
        pip install synapseclient
        synapse login
        synapse get syn4993293 --downloadLocation data/mpower
5. Ensure the directory structure matches:
       data/mpower/mPowerRelease1/mPower.csv
       data/mpower/mPowerRelease1/audio_video_responses/

Note: This dataset is subject to the Sage Bionetworks DUC.
      Do not redistribute the data.  Cite appropriately.
=====================================================================
"""
        print(instructions)


# Import for type hints in _load_from_disk (via base_loader)
from data.loaders.utils import LabelEncoder as _LE  # noqa: F811, E402
