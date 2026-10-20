"""Report generation agent for Voice Biomarker Platform v2."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, List, Optional

from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

__version__ = "2.0.0"


class ReportAgent(BaseAgent):
    """Compile outputs from all upstream agents into a structured report.

    Produces a JSON-serialisable dictionary suitable for both API responses
    and PDF generation pipelines.
    """

    def __init__(self) -> None:
        super().__init__(name="report")

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _execute(
        self,
        preprocessing: Dict[str, Any],
        biomarkers: Dict[str, Any],
        diagnosis: Dict[str, Any],
        explainability: Dict[str, Any],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build the final structured report.

        Parameters
        ----------
        preprocessing : dict
            Output of :class:`PreprocessingAgent`.
        biomarkers : dict
            Output of :class:`BiomarkerAgent`.
        diagnosis : dict
            Output of :class:`DiagnosisAgent`.
        explainability : dict
            Output of :class:`ExplainabilityAgent`.

        Returns
        -------
        dict
            Complete report with sections for each agent's results.
        """
        metadata = self._build_metadata(preprocessing)
        audio_info = self._build_audio_section(preprocessing)
        biomarker_section = self._build_biomarker_section(biomarkers)
        diagnosis_section = self._build_diagnosis_section(diagnosis)
        explanation_section = self._build_explanation_section(explainability)
        summary = self._build_summary(diagnosis, biomarkers)

        report = {
            "report_metadata": metadata,
            "audio_analysis": audio_info,
            "biomarkers": biomarker_section,
            "diagnosis": diagnosis_section,
            "explanations": explanation_section,
            "summary": summary,
            "disclaimer": self._disclaimer(),
        }

        self.logger.info("Report generated: %s", metadata["report_id"])
        return report

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata(preprocessing: Dict[str, Any]) -> Dict[str, Any]:
        """Top-level report metadata."""
        now = datetime.datetime.now(datetime.timezone.utc)
        report_id = now.strftime("VB2-%Y%m%d-%H%M%S") + "-" + __version__
        meta = preprocessing.get("metadata", {})

        return {
            "report_id": report_id,
            "platform_version": __version__,
            "generated_at": now.isoformat(),
            "filename": meta.get("filename", "unknown"),
            "duration_seconds": meta.get("duration", 0),
            "sample_rate": meta.get("sample_rate", 0),
        }

    @staticmethod
    def _build_audio_section(preprocessing: Dict[str, Any]) -> Dict[str, Any]:
        """Audio quality and preprocessing details."""
        quality = preprocessing.get("quality_metrics", {})
        meta = preprocessing.get("metadata", {})
        features = preprocessing.get("features", {})

        return {
            "quality": quality,
            "preprocessing_metadata": {
                "format": meta.get("format"),
                "original_duration": meta.get("duration"),
                "sample_rate": meta.get("sample_rate"),
                "num_features_extracted": len(features),
            },
            "validation_report": preprocessing.get("validation_report", ""),
        }

    @staticmethod
    def _build_biomarker_section(biomarkers: Dict[str, Any]) -> Dict[str, Any]:
        """Biomarker scores and anomaly details."""
        return {
            "parkinsons": biomarkers.get("parkinsons", {}),
            "depression": biomarkers.get("depression", {}),
            "anomalies": biomarkers.get("anomalies", []),
            "anomaly_details": biomarkers.get("anomaly_details", []),
        }

    @staticmethod
    def _build_diagnosis_section(diagnosis: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnosis predictions and recommendations."""
        return {
            "parkinsons": diagnosis.get("parkinsons", {}),
            "depression": diagnosis.get("depression", {}),
            "escalation_needed": diagnosis.get("escalation_needed", False),
            "recommendations": diagnosis.get("recommendations", []),
        }

    @staticmethod
    def _build_explanation_section(explainability: Dict[str, Any]) -> Dict[str, Any]:
        """Model explanation details."""
        return {
            "top_features": explainability.get("top_features", []),
            "feature_importance": explainability.get("feature_importance"),
            "shap_available": explainability.get("shap_explanation") is not None,
            "lime_available": explainability.get("lime_explanation") is not None,
            "plain_language": explainability.get("plain_language_explanation", ""),
        }

    @staticmethod
    def _build_summary(
        diagnosis: Dict[str, Any],
        biomarkers: Dict[str, Any],
    ) -> Dict[str, Any]:
        """High-level summary for quick review."""
        pd = diagnosis.get("parkinsons", {})
        dep = diagnosis.get("depression", {})

        key_findings: List[str] = []

        if pd.get("risk_level") == "high":
            key_findings.append("Elevated Parkinson's disease risk detected")
        elif pd.get("risk_level") == "moderate":
            key_findings.append("Moderate Parkinson's indicators present")

        if dep.get("risk_level") == "high":
            key_findings.append("Elevated depression risk detected")
        elif dep.get("risk_level") == "moderate":
            key_findings.append("Moderate depression indicators present")

        if diagnosis.get("escalation_needed"):
            key_findings.append("Clinical escalation recommended")

        if not key_findings:
            key_findings.append("No significant anomalies detected")

        pd_anomalies = biomarkers.get("parkinsons", {}).get("num_anomalies", 0)
        dep_anomalies = biomarkers.get("depression", {}).get("num_anomalies", 0)

        return {
            "key_findings": key_findings,
            "overall_risk_level": _max_risk(
                pd.get("risk_level", "low"), dep.get("risk_level", "low"),
            ),
            "parkinsons_risk": pd.get("risk_level", "unknown"),
            "depression_risk": dep.get("risk_level", "unknown"),
            "total_anomalies": pd_anomalies + dep_anomalies,
            "escalation_needed": diagnosis.get("escalation_needed", False),
        }

    @staticmethod
    def _disclaimer() -> str:
        return (
            "This report is generated by an automated voice biomarker analysis system "
            "and is intended for screening purposes only. It does not constitute a "
            "medical diagnosis. Results should be reviewed and interpreted by a "
            "qualified healthcare professional. The system uses acoustic analysis "
            "and machine learning models that have inherent limitations in accuracy "
            "and generalizability."
        )

    # ------------------------------------------------------------------
    # PDF-ready formatting
    # ------------------------------------------------------------------

    def to_pdf_payload(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Restructure the report for PDF template rendering.

        Flattens nested structures and adds section titles.
        """
        meta = report.get("report_metadata", {})
        summary = report.get("summary", {})
        diagnosis = report.get("diagnosis", {})
        biomarkers = report.get("biomarkers", {})
        explanations = report.get("explanations", {})
        audio = report.get("audio_analysis", {})

        return {
            "title": "Voice Biomarker Analysis Report",
            "subtitle": f"Report ID: {meta.get('report_id', 'N/A')}",
            "date": meta.get("generated_at", ""),
            "patient_file": meta.get("filename", ""),
            "sections": [
                {
                    "title": "Summary",
                    "content": {
                        "overall_risk": summary.get("overall_risk_level", "unknown"),
                        "key_findings": summary.get("key_findings", []),
                        "escalation": summary.get("escalation_needed", False),
                    },
                },
                {
                    "title": "Audio Quality",
                    "content": {
                        "quality_score": audio.get("quality", {}).get("quality_score", 0),
                        "snr_db": audio.get("quality", {}).get("snr_db", 0),
                        "duration": audio.get("preprocessing_metadata", {}).get("original_duration", 0),
                        "features_extracted": audio.get("preprocessing_metadata", {}).get("num_features_extracted", 0),
                    },
                },
                {
                    "title": "Parkinson's Disease Screening",
                    "content": {
                        "prediction": diagnosis.get("parkinsons", {}).get("prediction", "unknown"),
                        "risk_score": diagnosis.get("parkinsons", {}).get("risk_score", 0),
                        "risk_level": diagnosis.get("parkinsons", {}).get("risk_level", "unknown"),
                        "confidence": diagnosis.get("parkinsons", {}).get("confidence", 0),
                        "anomalies": biomarkers.get("parkinsons", {}).get("num_anomalies", 0),
                    },
                },
                {
                    "title": "Depression Screening",
                    "content": {
                        "prediction": diagnosis.get("depression", {}).get("prediction", "unknown"),
                        "risk_score": diagnosis.get("depression", {}).get("risk_score", 0),
                        "risk_level": diagnosis.get("depression", {}).get("risk_level", "unknown"),
                        "confidence": diagnosis.get("depression", {}).get("confidence", 0),
                        "anomalies": biomarkers.get("depression", {}).get("num_anomalies", 0),
                    },
                },
                {
                    "title": "Feature Analysis",
                    "content": {
                        "top_features": explanations.get("top_features", []),
                        "plain_language": explanations.get("plain_language", ""),
                    },
                },
                {
                    "title": "Recommendations",
                    "content": {
                        "items": diagnosis.get("recommendations", []),
                    },
                },
                {
                    "title": "Disclaimer",
                    "content": report.get("disclaimer", ""),
                },
            ],
        }


def _max_risk(a: str, b: str) -> str:
    """Return the higher of two risk level strings."""
    order = {"low": 0, "moderate": 1, "high": 2, "unknown": -1}
    return a if order.get(a, -1) >= order.get(b, -1) else b
