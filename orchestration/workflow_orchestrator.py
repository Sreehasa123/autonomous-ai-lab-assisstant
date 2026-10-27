"""Workflow Orchestrator."""
import json
import logging
from typing import Dict, Any, Optional

from database.repository import ExperimentRepository
from database.models import Experiment
from orchestration.workflow_state import WorkflowStateMachine, WorkflowStep
from schemas.experiment_schema import ExperimentStatus
from agents import (
    ProblemUnderstandingAgent,
    ExperimentPlanningAgent,
    DatasetSearchAgent,
    DatasetComparisonAgent,
    DatasetAnalysisAgent,
    DataQualityAgent,
    PreprocessingAgent,
    ModelSelectionAgent,
    CodeGenerationAgent,
    CodeValidationAgent,
    ExperimentExecutionAgent,
    EvaluationAgent,
    ErrorAnalysisAgent,
    ImprovementAgent,
    ReportGenerationAgent
)

logger = logging.getLogger(__name__)

class WorkflowOrchestrator:
    """Coordinates agent execution, manages persistent workflow state, and handles human-in-the-loop approvals."""

    def __init__(self, repo: Optional[ExperimentRepository] = None):
        self.repo = repo or ExperimentRepository()
        
        # Instantiate Agents
        self.problem_agent = ProblemUnderstandingAgent()
        self.planning_agent = ExperimentPlanningAgent()
        self.search_agent = DatasetSearchAgent()
        self.comparison_agent = DatasetComparisonAgent()
        self.analysis_agent = DatasetAnalysisAgent()
        self.quality_agent = DataQualityAgent()
        self.preprocessing_agent = PreprocessingAgent()
        self.model_agent = ModelSelectionAgent()
        self.code_gen_agent = CodeGenerationAgent()
        self.code_val_agent = CodeValidationAgent()
        self.execution_agent = ExperimentExecutionAgent()
        self.evaluation_agent = EvaluationAgent()
        self.error_agent = ErrorAnalysisAgent()
        self.improvement_agent = ImprovementAgent()
        self.report_agent = ReportGenerationAgent()

    def process_problem_formulation(self, experiment_id: str) -> Dict[str, Any]:
        """Runs problem understanding and initial experiment planning."""
        exp = self.repo.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Step 1: Problem Understanding
        res_understanding = self.problem_agent.run({"problem_statement": exp.problem_statement})
        understanding = res_understanding.get("problem_understanding", {})

        # Step 2: Experiment Planning
        res_plan = self.planning_agent.run({"problem_understanding": understanding})
        plan = res_plan.get("experiment_plan", {})

        # Update in DB
        self.repo.update_experiment(
            experiment_id=experiment_id,
            status=ExperimentStatus.PLANNING.value,
            current_step=WorkflowStep.EXPERIMENT_PLANNING.value,
            problem_understanding_json=json.dumps(understanding),
            experiment_plan_json=json.dumps(plan)
        )
        return {"problem_understanding": understanding, "experiment_plan": plan}

    def search_and_rank_datasets(self, experiment_id: str, search_query: Optional[str] = None) -> Dict[str, Any]:
        """Discovers candidate datasets and ranks them by suitability."""
        exp = self.repo.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        understanding = json.loads(exp.problem_understanding_json or "{}")
        q = search_query or exp.problem_statement

        # Step 3: Dataset Search
        search_output = self.search_agent.run({"search_query": q, "problem_statement": exp.problem_statement})
        candidates = search_output.get("datasets", [])

        # Step 4: Dataset Comparison & Ranking
        ranking_output = self.comparison_agent.run({
            "datasets": candidates,
            "problem_type": understanding.get("problem_type", "tabular_classification")
        })

        self.repo.update_experiment(
            experiment_id=experiment_id,
            status=ExperimentStatus.SEARCHING_DATASETS.value,
            current_step=WorkflowStep.DATASET_SELECTION.value
        )
        return {
            "datasets": ranking_output.get("ranked_datasets", []),
            "providers": search_output.get("providers", {}),
            "total_found": search_output.get("total_found", 0)
        }

    def select_and_analyze_dataset(self, experiment_id: str, dataset_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Selects dataset, runs statistical profiling, and audits data quality."""
        exp = self.repo.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        # Step 5: Dataset Analysis
        analysis_res = self.analysis_agent.run({"selected_dataset": dataset_meta})
        analysis_data = analysis_res.get("analysis_result", {})

        # Step 6: Data Quality Audit
        quality_res = self.quality_agent.run({"analysis_result": analysis_data})

        # Step 7: Preprocessing Proposal
        prep_res = self.preprocessing_agent.run({
            "analysis_result": analysis_data,
            "problem_type": dataset_meta.get("task_type", "tabular_classification")
        })

        # Step 8: Model Recommendations
        model_res = self.model_agent.run({
            "analysis_result": analysis_data,
            "problem_type": dataset_meta.get("task_type", "tabular_classification"),
            "is_deep_learning": False
        })

        self.repo.update_experiment(
            experiment_id=experiment_id,
            status=ExperimentStatus.DATASET_SELECTED.value,
            current_step=WorkflowStep.DATASET_ANALYSIS.value,
            selected_dataset_json=json.dumps(dataset_meta),
            dataset_analysis_json=json.dumps(analysis_data),
            preprocessing_config_json=json.dumps(prep_res.get("preprocessing_config", {})),
            model_config_json=json.dumps(model_res.get("default_model", {}))
        )

        return {
            "selected_dataset": dataset_meta,
            "analysis_result": analysis_data,
            "quality_audit": quality_res,
            "preprocessing_config": prep_res.get("preprocessing_config"),
            "recommended_models": model_res.get("recommended_models")
        }

    def generate_and_validate_code(self, experiment_id: str) -> Dict[str, Any]:
        """Synthesizes experiment code and executes AST syntax validation."""
        exp = self.repo.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        dataset_meta = json.loads(exp.selected_dataset_json or "{}")
        prep_config = json.loads(exp.preprocessing_config_json or "{}")
        model_config = json.loads(exp.model_config_json or "{}")

        # Code Generation
        gen_res = self.code_gen_agent.run({
            "experiment_id": exp.id,
            "selected_dataset": dataset_meta,
            "preprocessing_config": prep_config,
            "model_config": model_config
        })
        code = gen_res.get("generated_code", "")

        # Code Validation
        val_res = self.code_val_agent.run({"generated_code": code})
        validation = val_res.get("validation_result", {})

        new_status = ExperimentStatus.READY_TO_RUN.value if validation.get("is_valid") else ExperimentStatus.FAILED.value
        self.repo.update_experiment(
            experiment_id=experiment_id,
            status=new_status,
            current_step=WorkflowStep.CODE_VALIDATION.value,
            last_validation_json=json.dumps(validation)
        )

        return {"generated_code": code, "validation_result": validation}
