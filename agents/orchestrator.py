"""Pipeline orchestrator for Voice Biomarker Platform v2.

Coordinates the full analysis pipeline:

    preprocess → biomarkers → diagnosis → explainability → report
"""

from __future__ import annotations

import asyncio
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from agents.base_agent import AgentStatus
from agents.biomarker_agent import BiomarkerAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.preprocessing_agent import PreprocessingAgent
from agents.report_agent import ReportAgent

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when a pipeline stage fails irrecoverably."""

    def __init__(self, stage: str, message: str, cause: Optional[Exception] = None):
        self.stage = stage
        self.cause = cause
        super().__init__(f"Pipeline failed at stage '{stage}': {message}")


class AgentOrchestrator:
    """Coordinate all agents through the analysis pipeline.

    Parameters
    ----------
    quality_threshold : float, optional
        Minimum audio quality score for the preprocessing stage.
    checkpoint_dir : str or Path, optional
        Directory containing model checkpoints.
    use_deep_model : bool, optional
        Force deep model usage in diagnosis.
    """

    def __init__(
        self,
        quality_threshold: Optional[float] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        use_deep_model: Optional[bool] = None,
    ) -> None:
        # Instantiate all agents
        self.preprocessing_agent = PreprocessingAgent(
            quality_threshold=quality_threshold,
        )
        self.biomarker_agent = BiomarkerAgent()
        self.diagnosis_agent = DiagnosisAgent(
            checkpoint_dir=checkpoint_dir,
            use_deep_model=use_deep_model,
        )
        self.explainability_agent = ExplainabilityAgent()
        self.report_agent = ReportAgent()

        self._agents = [
            self.preprocessing_agent,
            self.biomarker_agent,
            self.diagnosis_agent,
            self.explainability_agent,
            self.report_agent,
        ]

        self._stage_timings: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def agents(self) -> List:
        """All registered agents."""
        return list(self._agents)

    @property
    def total_duration(self) -> float:
        """Sum of all stage durations from the last run."""
        return sum(self._stage_timings.values())

    # ------------------------------------------------------------------
    # Synchronous pipeline
    # ------------------------------------------------------------------

    def run(self, filepath: Union[str, Path], **kwargs: Any) -> Dict[str, Any]:
        """Execute the full analysis pipeline synchronously.

        Parameters
        ----------
        filepath : str or Path
            Path to the audio file to analyse.

        Returns
        -------
        dict
            Complete report with all agent results.

        Raises
        ------
        PipelineError
            If any stage fails and recovery is not possible.
        """
        filepath = Path(filepath)
        self._stage_timings.clear()
        start = time.perf_counter()

        logger.info("Starting pipeline for %s", filepath.name)

        # ---- Stage 1: Preprocessing ----
        preprocessing_result = self._run_stage(
            "preprocessing", self.preprocessing_agent, filepath=filepath,
        )

        # ---- Stage 2: Biomarker analysis ----
        biomarker_result = self._run_stage(
            "biomarker",
            self.biomarker_agent,
            features=preprocessing_result["features"],
        )

        # ---- Stage 3: Diagnosis ----
        diagnosis_result = self._run_stage(
            "diagnosis",
            self.diagnosis_agent,
            features=preprocessing_result["features"],
            biomarkers=biomarker_result,
            audio=preprocessing_result["audio"],
        )

        # ---- Stage 4: Explainability ----
        explanation_result = self._run_stage(
            "explainability",
            self.explainability_agent,
            features=preprocessing_result["features"],
            feature_names=self._get_feature_names(preprocessing_result),
            diagnosis=diagnosis_result,
        )

        # ---- Stage 5: Report ----
        report = self._run_stage(
            "report",
            self.report_agent,
            preprocessing=preprocessing_result,
            biomarkers=biomarker_result,
            diagnosis=diagnosis_result,
            explainability=explanation_result,
        )

        total = time.perf_counter() - start
        report["pipeline_timing"] = {
            "total_seconds": round(total, 4),
            "stage_seconds": {
                k: round(v, 4) for k, v in self._stage_timings.items()
            },
        }

        logger.info("Pipeline complete in %.3fs for %s", total, filepath.name)
        return report

    # ------------------------------------------------------------------
    # Async pipeline
    # ------------------------------------------------------------------

    async def run_async(self, filepath: Union[str, Path], **kwargs: Any) -> Dict[str, Any]:
        """Execute the full pipeline asynchronously using a thread pool.

        Each stage is dispatched to a ``ThreadPoolExecutor`` so the event
        loop is not blocked by CPU-bound work.
        """
        filepath = Path(filepath)
        self._stage_timings.clear()
        start = time.perf_counter()

        logger.info("Starting async pipeline for %s", filepath.name)

        loop = asyncio.get_running_loop()

        # Stage 1: Preprocessing
        preprocessing_result = await loop.run_in_executor(
            None, lambda: self._run_stage(
                "preprocessing", self.preprocessing_agent, filepath=filepath,
            ),
        )

        # Stage 2: Biomarkers
        biomarker_result = await loop.run_in_executor(
            None, lambda: self._run_stage(
                "biomarker", self.biomarker_agent,
                features=preprocessing_result["features"],
            ),
        )

        # Stage 3: Diagnosis
        diagnosis_result = await loop.run_in_executor(
            None, lambda: self._run_stage(
                "diagnosis", self.diagnosis_agent,
                features=preprocessing_result["features"],
                biomarkers=biomarker_result,
                audio=preprocessing_result["audio"],
            ),
        )

        # Stage 4: Explainability
        explanation_result = await loop.run_in_executor(
            None, lambda: self._run_stage(
                "explainability", self.explainability_agent,
                features=preprocessing_result["features"],
                feature_names=self._get_feature_names(preprocessing_result),
                diagnosis=diagnosis_result,
            ),
        )

        # Stage 5: Report
        report = await loop.run_in_executor(
            None, lambda: self._run_stage(
                "report", self.report_agent,
                preprocessing=preprocessing_result,
                biomarkers=biomarker_result,
                diagnosis=diagnosis_result,
                explainability=explanation_result,
            ),
        )

        total = time.perf_counter() - start
        report["pipeline_timing"] = {
            "total_seconds": round(total, 4),
            "stage_seconds": {
                k: round(v, 4) for k, v in self._stage_timings.items()
            },
        }

        logger.info("Async pipeline complete in %.3fs for %s", total, filepath.name)
        return report

    # ------------------------------------------------------------------
    # Partial / individual stage runners
    # ------------------------------------------------------------------

    def run_preprocessing_only(self, filepath: Union[str, Path]) -> Dict[str, Any]:
        """Run only the preprocessing stage."""
        return self._run_stage(
            "preprocessing", self.preprocessing_agent, filepath=filepath,
        )

    def run_biomarkers_only(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Run only the biomarker analysis stage."""
        return self._run_stage(
            "biomarker", self.biomarker_agent, features=features,
        )

    def run_diagnosis_only(
        self,
        features: Dict[str, Any],
        biomarkers: Optional[Dict[str, Any]] = None,
        audio: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run only the diagnosis stage."""
        if biomarkers is None:
            biomarkers = self.run_biomarkers_only(features)
        return self._run_stage(
            "diagnosis", self.diagnosis_agent,
            features=features, biomarkers=biomarkers, audio=audio,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return the status of all agents."""
        return {
            "agents": [agent.get_info() for agent in self._agents],
            "last_timing": {
                "total": round(self.total_duration, 4),
                "stages": {
                    k: round(v, 4) for k, v in self._stage_timings.items()
                },
            },
        }

    def reset(self) -> None:
        """Reset all agents to idle state."""
        for agent in self._agents:
            agent.reset()
        self._stage_timings.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_stage(
        self,
        stage_name: str,
        agent: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute a single pipeline stage with timing and error handling."""
        start = time.perf_counter()
        try:
            result = agent.run(**kwargs)
            elapsed = time.perf_counter() - start
            self._stage_timings[stage_name] = elapsed
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - start
            self._stage_timings[stage_name] = elapsed
            raise PipelineError(
                stage=stage_name,
                message=str(exc),
                cause=exc,
            ) from exc

    @staticmethod
    def _get_feature_names(preprocessing_result: Dict[str, Any]) -> List[str]:
        """Extract ordered scalar feature names from preprocessing output."""
        features = preprocessing_result.get("features", {})
        return sorted(k for k, v in features.items() if isinstance(v, (int, float)))
