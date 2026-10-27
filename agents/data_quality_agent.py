"""Data Quality and Leakage Audit Agent."""
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from schemas.dataset_schema import DataQualityFinding

class DataQualityAgent(BaseAgent):
    """Audits dataset for integrity, leakage, imbalance, and anomalies."""

    def __init__(self):
        super().__init__(
            name="Data Quality Agent",
            description="Examines data health, detects leakage vectors, and classifies risks."
        )

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        analysis_dict = context.get("analysis_result", {})
        findings = analysis_dict.get("findings", [])

        # Categorize by severity
        critical = [f for f in findings if f.get("severity") == "critical"]
        high_risk = [f for f in findings if f.get("severity") == "high_risk"]
        warnings = [f for f in findings if f.get("severity") == "warning"]
        info = [f for f in findings if f.get("severity") == "info"]

        has_blocker = len(critical) > 0
        status = "PASSED"
        if critical:
            status = "CRITICAL_ISSUES"
        elif high_risk:
            status = "HIGH_RISK_WARNINGS"
        elif warnings:
            status = "WARNINGS_PRESENT"

        return {
            "quality_status": status,
            "has_blocking_issues": has_blocker,
            "counts": {
                "critical": len(critical),
                "high_risk": len(high_risk),
                "warning": len(warnings),
                "info": len(info)
            },
            "findings": findings
        }
