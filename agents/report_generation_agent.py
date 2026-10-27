"""Report Generation Agent."""
from pathlib import Path
from typing import Dict, Any
from agents.base_agent import BaseAgent
from services.report_service import ReportService

class ReportGenerationAgent(BaseAgent):
    """Generates standardized experiment documentation in Markdown, HTML, or JSON."""

    def __init__(self):
        super().__init__(
            name="Report Generation Agent",
            description="Compiles verified data, code, metrics, and diagnostics into comprehensive laboratory reports."
        )
        self.report_service = ReportService()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        experiment_id = context.get("experiment_id")
        report_format = context.get("report_format", "markdown")
        if not experiment_id:
            raise ValueError("experiment_id is required to compile a report.")

        report_path = self.report_service.generate_report(
            experiment_id=experiment_id,
            report_format=report_format
        )
        return {
            "report_path": str(report_path),
            "report_format": report_format,
            "file_name": report_path.name
        }
