"""Voice Biomarker Platform v2 – agents package.

Provides a modular pipeline of analysis agents:

* :class:`~agents.preprocessing_agent.PreprocessingAgent`
* :class:`~agents.biomarker_agent.BiomarkerAgent`
* :class:`~agents.diagnosis_agent.DiagnosisAgent`
* :class:`~agents.explainability_agent.ExplainabilityAgent`
* :class:`~agents.report_agent.ReportAgent`
* :class:`~agents.orchestrator.AgentOrchestrator`

Usage::

    from agents import AgentOrchestrator

    orchestrator = AgentOrchestrator()
    report = orchestrator.run("recording.wav")
"""

from agents.base_agent import AgentStatus, BaseAgent
from agents.biomarker_agent import BiomarkerAgent
from agents.diagnosis_agent import DiagnosisAgent
from agents.explainability_agent import ExplainabilityAgent
from agents.orchestrator import AgentOrchestrator, PipelineError
from agents.preprocessing_agent import AudioQualityError, PreprocessingAgent
from agents.report_agent import ReportAgent

__all__ = [
    # Base
    "BaseAgent",
    "AgentStatus",
    # Agents
    "PreprocessingAgent",
    "BiomarkerAgent",
    "DiagnosisAgent",
    "ExplainabilityAgent",
    "ReportAgent",
    # Orchestrator
    "AgentOrchestrator",
    "PipelineError",
    # Errors
    "AudioQualityError",
]
