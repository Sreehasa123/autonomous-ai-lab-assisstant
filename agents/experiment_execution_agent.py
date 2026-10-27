"""Experiment Execution Agent."""
from typing import Dict, Any
from agents.base_agent import BaseAgent
from services.experiment_service import ExperimentService

class ExperimentExecutionAgent(BaseAgent):
    """Supervises and executes training sessions safely, tracking logs and status."""

    def __init__(self):
        super().__init__(
            name="Experiment Execution Agent",
            description="Manages local and external training execution runs, recording duration and state changes."
        )
        self.experiment_service = ExperimentService()

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        experiment_id = context.get("experiment_id")
        if not experiment_id:
            raise ValueError("experiment_id is required for execution.")

        success, msg, run_record = self.experiment_service.execute_local_training(experiment_id)
        return {
            "success": success,
            "message": msg,
            "run_id": run_record.id if run_record else None,
            "metrics": json.loads(run_record.metrics_json) if run_record and run_record.metrics_json else {}
        }
import json
