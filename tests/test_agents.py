"""Tests for agents module: each agent individually, orchestrator, error handling."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.base_agent import BaseAgent, AgentStatus
from agents.preprocessing_agent import PreprocessingAgent, AudioQualityError
from agents.biomarker_agent import BiomarkerAgent, PARKINSONS_THRESHOLDS, DEPRESSION_THRESHOLDS
from agents.diagnosis_agent import DiagnosisAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.report_agent import ReportAgent
from agents.orchestrator import AgentOrchestrator, PipelineError


# ======================================================================
# BaseAgent
# ======================================================================

class TestBaseAgent:
    """Test BaseAgent abstract class."""

    def test_init(self) -> None:
        class DummyAgent(BaseAgent):
            def _execute(self, **kwargs):
                return {"result": "ok"}

        agent = DummyAgent(name="test")
        assert agent.name == "test"
        assert agent.status == AgentStatus.IDLE
        assert agent.last_error is None

    def test_run_success(self) -> None:
        class DummyAgent(BaseAgent):
            def _execute(self, **kwargs):
                return {"result": "ok"}

        agent = DummyAgent(name="test")
        result = agent.run()
        assert result == {"result": "ok"}
        assert agent.status == AgentStatus.COMPLETED
        assert agent._run_count == 1

    def test_run_failure(self) -> None:
        class FailingAgent(BaseAgent):
            def _execute(self, **kwargs):
                raise ValueError("boom")

        agent = FailingAgent(name="failing")
        with pytest.raises(ValueError, match="boom"):
            agent.run()
        assert agent.status == AgentStatus.FAILED
        assert "boom" in agent.last_error

    def test_get_info(self) -> None:
        class DummyAgent(BaseAgent):
            def _execute(self, **kwargs):
                return {}

        agent = DummyAgent(name="test")
        agent.run()
        info = agent.get_info()
        assert info["name"] == "test"
        assert info["status"] == "completed"
        assert info["run_count"] == 1
        assert info["average_duration"] >= 0

    def test_reset(self) -> None:
        class DummyAgent(BaseAgent):
            def _execute(self, **kwargs):
                raise RuntimeError("fail")

        agent = DummyAgent(name="test")
        try:
            agent.run()
        except RuntimeError:
            pass
        assert agent.status == AgentStatus.FAILED
        agent.reset()
        assert agent.status == AgentStatus.IDLE
        assert agent.last_error is None

    def test_repr(self) -> None:
        class DummyAgent(BaseAgent):
            def _execute(self, **kwargs):
                return {}

        agent = DummyAgent(name="test")
        r = repr(agent)
        assert "DummyAgent" in r
        assert "test" in r


# ======================================================================
# BiomarkerAgent
# ======================================================================

class TestBiomarkerAgent:
    """Test biomarker analysis."""

    def test_run_healthy(self, sample_features: Dict[str, float]) -> None:
        agent = BiomarkerAgent()
        result = agent.run(features=sample_features)
        assert "parkinsons" in result
        assert "depression" in result
        assert "anomalies" in result
        assert "biomarker_scores" in result
        assert 0 <= result["biomarker_scores"]["parkinsons_risk"] <= 1
        assert 0 <= result["biomarker_scores"]["depression_risk"] <= 1

    def test_run_anomalous(self, anomalous_features: Dict[str, float]) -> None:
        agent = BiomarkerAgent()
        result = agent.run(features=anomalous_features)
        assert len(result["anomalies"]) > 0
        assert result["parkinsons"]["risk_level"] in ("moderate", "high")
        assert result["depression"]["risk_level"] in ("moderate", "high")

    def test_anomaly_detection_directions(self, anomalous_features: Dict[str, float]) -> None:
        agent = BiomarkerAgent()
        result = agent.run(features=anomalous_features)
        for anomaly in result["anomalies"]:
            assert "direction" in anomaly
            assert anomaly["direction"] in ("high", "low")

    def test_anomaly_details(self, anomalous_features: Dict[str, float]) -> None:
        agent = BiomarkerAgent()
        result = agent.run(features=anomalous_features)
        assert len(result["anomaly_details"]) > 0
        for detail in result["anomaly_details"]:
            assert "explanation" in detail
            assert len(detail["explanation"]) > 0


# ======================================================================
# DiagnosisAgent
# ======================================================================

class TestDiagnosisAgent:
    """Test diagnosis agent."""

    def test_rule_based_predict(
        self, sample_features: Dict[str, float],
        sample_biomarker_result: Dict[str, Any],
    ) -> None:
        with patch("config.settings.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_settings.model.checkpoint_dir = "models/checkpoints"
            mock_settings.model.use_deep_model = False
            mock_settings.model.model_type = "xgboost"
            mock_get.return_value = mock_settings

            agent = DiagnosisAgent()
            result = agent.run(
                features=sample_features,
                biomarkers=sample_biomarker_result,
            )

        assert "parkinsons" in result
        assert "depression" in result
        assert "escalation_needed" in result
        assert "recommendations" in result
        assert result["parkinsons"]["method"] == "rule_based"
        assert result["depression"]["method"] == "rule_based"

    def test_escalation_low_risk(
        self, sample_features: Dict[str, float],
        sample_biomarker_result: Dict[str, Any],
    ) -> None:
        with patch("config.settings.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_settings.model.checkpoint_dir = "models/checkpoints"
            mock_settings.model.use_deep_model = False
            mock_get.return_value = mock_settings

            agent = DiagnosisAgent()
            result = agent.run(
                features=sample_features,
                biomarkers=sample_biomarker_result,
            )

        assert result["escalation_needed"] is False

    def test_risk_levels(self) -> None:
        assert DiagnosisAgent._risk_level(0.8) == "high"
        assert DiagnosisAgent._risk_level(0.4) == "moderate"
        assert DiagnosisAgent._risk_level(0.1) == "low"


# ======================================================================
# ExplainabilityAgent
# ======================================================================

class TestExplainabilityAgent:
    """Test explainability agent."""

    def test_run_without_model(
        self, sample_features: Dict[str, float],
    ) -> None:
        with patch("config.settings.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_settings.explainability.num_features = 10
            mock_settings.explainability.use_shap = False
            mock_settings.explainability.use_lime = False
            mock_get.return_value = mock_settings

            agent = ExplainabilityAgent()
            result = agent.run(
                features=sample_features,
                feature_names=list(sample_features.keys()),
            )

        assert "shap_explanation" in result
        assert "lime_explanation" in result
        assert "feature_importance" in result
        assert "top_features" in result
        assert "plain_language_explanation" in result

    def test_empty_explanation(self) -> None:
        with patch("config.settings.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_settings.explainability.num_features = 10
            mock_settings.explainability.use_shap = False
            mock_settings.explainability.use_lime = False
            mock_get.return_value = mock_settings

            agent = ExplainabilityAgent()
            result = agent.run(features={}, feature_names=[])

        assert result["top_features"] == []
        assert "Insufficient" in result["plain_language_explanation"]


# ======================================================================
# ReportAgent
# ======================================================================

class TestReportAgent:
    """Test report generation."""

    def test_run(
        self,
        sample_preprocessing_result: Dict[str, Any],
        sample_biomarker_result: Dict[str, Any],
        sample_diagnosis_result: Dict[str, Any],
        sample_explainability_result: Dict[str, Any],
    ) -> None:
        agent = ReportAgent()
        result = agent.run(
            preprocessing=sample_preprocessing_result,
            biomarkers=sample_biomarker_result,
            diagnosis=sample_diagnosis_result,
            explainability=sample_explainability_result,
        )

        assert "report_metadata" in result
        assert "audio_analysis" in result
        assert "biomarkers" in result
        assert "diagnosis" in result
        assert "explanations" in result
        assert "summary" in result
        assert "disclaimer" in result
        assert "report_id" in result["report_metadata"]
        assert "VB2-" in result["report_metadata"]["report_id"]

    def test_pdf_payload(
        self,
        sample_preprocessing_result: Dict[str, Any],
        sample_biomarker_result: Dict[str, Any],
        sample_diagnosis_result: Dict[str, Any],
        sample_explainability_result: Dict[str, Any],
    ) -> None:
        agent = ReportAgent()
        report = agent.run(
            preprocessing=sample_preprocessing_result,
            biomarkers=sample_biomarker_result,
            diagnosis=sample_diagnosis_result,
            explainability=sample_explainability_result,
        )
        pdf_payload = agent.to_pdf_payload(report)
        assert "title" in pdf_payload
        assert "sections" in pdf_payload
        assert len(pdf_payload["sections"]) > 0

    def test_disclaimer(self) -> None:
        agent = ReportAgent()
        disclaimer = agent._disclaimer()
        assert "screening" in disclaimer.lower() or "Screening" in disclaimer
        assert "healthcare" in disclaimer.lower() or "healthcare" in disclaimer


# ======================================================================
# PreprocessingAgent
# ======================================================================

class TestPreprocessingAgent:
    """Test preprocessing agent."""

    @pytest.fixture(autouse=True)
    def _mock_settings(self):
        with patch("config.settings.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_settings.audio.target_sr = 16000
            mock_settings.audio.quality_threshold = 0.0
            mock_settings.audio.min_duration = 0.5
            mock_settings.audio.max_duration = 30.0
            mock_get.return_value = mock_settings
            yield

    def test_run(self, sample_wav_path: Path) -> None:
        try:
            import parselmouth  # noqa: F401
        except ImportError:
            pytest.skip("parselmouth not available or broken on this Python version")
        agent = PreprocessingAgent(quality_threshold=0.0)
        result = agent.run(filepath=sample_wav_path)
        assert "audio" in result
        assert "features" in result
        assert "quality_metrics" in result
        assert "metadata" in result
        assert isinstance(result["audio"], np.ndarray)

    def test_feature_vector(self, sample_features: Dict[str, float]) -> None:
        with patch("config.settings.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_settings.audio.target_sr = 16000
            mock_settings.audio.quality_threshold = 0.0
            mock_settings.audio.min_duration = 0.5
            mock_settings.audio.max_duration = 30.0
            mock_get.return_value = mock_settings
            agent = PreprocessingAgent(quality_threshold=0.0)

        vec = agent.get_feature_vector(sample_features)
        assert vec.ndim == 2
        assert vec.shape[0] == 1

    def test_feature_names(self, sample_features: Dict[str, float]) -> None:
        with patch("config.settings.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_settings.audio.target_sr = 16000
            mock_settings.audio.quality_threshold = 0.0
            mock_settings.audio.min_duration = 0.5
            mock_settings.audio.max_duration = 30.0
            mock_get.return_value = mock_settings
            agent = PreprocessingAgent(quality_threshold=0.0)

        names = agent.get_feature_names(sample_features)
        assert isinstance(names, list)
        assert len(names) > 0

    def test_quality_rejection(self, sample_wav_path: Path) -> None:
        with patch("config.settings.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_settings.audio.target_sr = 16000
            mock_settings.audio.quality_threshold = 0.99
            mock_settings.audio.min_duration = 0.5
            mock_settings.audio.max_duration = 30.0
            mock_get.return_value = mock_settings

            agent = PreprocessingAgent(quality_threshold=0.99)
            with pytest.raises(AudioQualityError):
                agent.run(filepath=sample_wav_path)


# ======================================================================
# AgentOrchestrator
# ======================================================================

class TestAgentOrchestrator:
    """Test the full pipeline orchestrator."""

    @pytest.fixture(autouse=True)
    def _mock_all_settings(self):
        with patch("config.settings.get_settings") as mock_get:
            mock_settings = MagicMock()
            mock_settings.audio.target_sr = 16000
            mock_settings.audio.quality_threshold = 0.0
            mock_settings.audio.min_duration = 0.5
            mock_settings.audio.max_duration = 30.0
            mock_settings.model.checkpoint_dir = "models/checkpoints"
            mock_settings.model.use_deep_model = False
            mock_settings.model.model_type = "xgboost"
            mock_settings.explainability.num_features = 10
            mock_settings.explainability.use_shap = False
            mock_settings.explainability.use_lime = False
            mock_get.return_value = mock_settings
            yield

    def test_init(self) -> None:
        orch = AgentOrchestrator()
        assert len(orch.agents) == 5
        assert orch.total_duration == 0.0

    def test_agents_list(self) -> None:
        orch = AgentOrchestrator()
        names = [a.name for a in orch.agents]
        assert "preprocessing" in names
        assert "biomarker" in names
        assert "diagnosis" in names
        assert "explainability" in names
        assert "report" in names

    def test_get_status(self) -> None:
        orch = AgentOrchestrator()
        status = orch.get_status()
        assert "agents" in status
        assert len(status["agents"]) == 5

    def test_reset(self) -> None:
        orch = AgentOrchestrator()
        orch.reset()
        for agent in orch.agents:
            assert agent.status == AgentStatus.IDLE

    def test_preprocessing_failure(self, sample_wav_path: Path) -> None:
        orch = AgentOrchestrator(quality_threshold=0.99)
        with pytest.raises(PipelineError) as exc_info:
            orch.run(sample_wav_path)
        assert "preprocessing" in str(exc_info.value.stage)

    def test_pipeline_error_attributes(self) -> None:
        err = PipelineError(stage="test", message="boom", cause=ValueError("x"))
        assert err.stage == "test"
        assert str(err) == "Pipeline failed at stage 'test': boom"
