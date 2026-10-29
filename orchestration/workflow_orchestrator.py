"""Workflow Orchestrator — Upgraded with dynamic dataset-free branching."""
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
    ReportGenerationAgent,
)

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """
    Coordinates agent execution, manages persistent workflow state,
    and handles both dataset-required and dataset-free experiment workflows.
    """

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

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 1: Problem Formulation (shared by all paths)
    # ─────────────────────────────────────────────────────────────────────────

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
            experiment_plan_json=json.dumps(plan),
        )
        return {"problem_understanding": understanding, "experiment_plan": plan}

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2A: Dataset Search (only for dataset-required experiments)
    # ─────────────────────────────────────────────────────────────────────────

    def search_and_rank_datasets(
        self, experiment_id: str, search_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """Discovers candidate datasets and ranks them by suitability."""
        exp = self.repo.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        understanding = json.loads(exp.problem_understanding_json or "{}")
        q = search_query or exp.problem_statement

        search_output = self.search_agent.run(
            {"search_query": q, "problem_statement": exp.problem_statement}
        )
        candidates = search_output.get("datasets", [])

        ranking_output = self.comparison_agent.run(
            {
                "datasets": candidates,
                "problem_type": understanding.get("problem_type", "tabular_classification"),
            }
        )

        self.repo.update_experiment(
            experiment_id=experiment_id,
            status=ExperimentStatus.SEARCHING_DATASETS.value,
            current_step=WorkflowStep.DATASET_SELECTION.value,
        )
        return {
            "datasets": ranking_output.get("ranked_datasets", []),
            "providers": search_output.get("providers", {}),
            "total_found": search_output.get("total_found", 0),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 2B: Dataset-Free Code Generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_code_without_dataset(self, experiment_id: str) -> Dict[str, Any]:
        """
        Generates code for experiments that don't require an external dataset
        (gradient descent, activation comparison, K-means demo, etc.)
        """
        exp = self.repo.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        understanding = json.loads(exp.problem_understanding_json or "{}")

        self.repo.update_experiment(
            experiment_id=experiment_id,
            status=ExperimentStatus.DATASET_NOT_REQUIRED.value,
            current_step=WorkflowStep.CODE_GENERATION.value,
        )

        # Generate problem-specific code
        gen_res = self.code_gen_agent.run(
            {
                "experiment_id": exp.id,
                "problem_statement": exp.problem_statement,
                "problem_understanding": understanding,
                "selected_dataset": {},
                "preprocessing_config": {},
                "model_config": {},
            }
        )
        code = gen_res.get("generated_code", "")

        # Validate the generated code
        val_res = self.code_val_agent.run({"generated_code": code})
        validation = val_res.get("validation_result", {})

        new_status = (
            ExperimentStatus.READY_TO_RUN.value
            if validation.get("is_valid")
            else ExperimentStatus.GENERATING_CODE.value
        )
        self.repo.update_experiment(
            experiment_id=experiment_id,
            status=new_status,
            current_step=WorkflowStep.CODE_VALIDATION.value,
            last_validation_json=json.dumps(validation),
        )

        return {
            "generated_code": code,
            "validation_result": validation,
            "dataset_required": False,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 3: Dataset Analysis (for dataset-required path)
    # ─────────────────────────────────────────────────────────────────────────

    def select_and_analyze_dataset(
        self, experiment_id: str, dataset_meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Selects dataset, runs statistical profiling, and audits data quality."""
        exp = self.repo.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        analysis_res = self.analysis_agent.run({"selected_dataset": dataset_meta})
        analysis_data = analysis_res.get("analysis_result", {})

        quality_res = self.quality_agent.run({"analysis_result": analysis_data})

        prep_res = self.preprocessing_agent.run(
            {
                "analysis_result": analysis_data,
                "problem_type": dataset_meta.get("task_type", "tabular_classification"),
            }
        )

        model_res = self.model_agent.run(
            {
                "analysis_result": analysis_data,
                "problem_type": dataset_meta.get("task_type", "tabular_classification"),
                "is_deep_learning": False,
            }
        )

        self.repo.update_experiment(
            experiment_id=experiment_id,
            status=ExperimentStatus.DATASET_SELECTED.value,
            current_step=WorkflowStep.DATASET_ANALYSIS.value,
            selected_dataset_json=json.dumps(dataset_meta),
            dataset_analysis_json=json.dumps(analysis_data),
            preprocessing_config_json=json.dumps(prep_res.get("preprocessing_config", {})),
            model_config_json=json.dumps(model_res.get("default_model", {})),
        )

        return {
            "selected_dataset": dataset_meta,
            "analysis_result": analysis_data,
            "quality_audit": quality_res,
            "preprocessing_config": prep_res.get("preprocessing_config"),
            "recommended_models": model_res.get("recommended_models"),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 4: Code Generation & Validation (dataset path)
    # ─────────────────────────────────────────────────────────────────────────

    def generate_and_validate_code(self, experiment_id: str) -> Dict[str, Any]:
        """Synthesizes experiment code and executes AST syntax validation."""
        exp = self.repo.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        understanding = json.loads(exp.problem_understanding_json or "{}")
        if not understanding.get("dataset_required", True):
            return self.generate_code_without_dataset(experiment_id)

        dataset_meta = json.loads(exp.selected_dataset_json or "{}")
        prep_config = json.loads(exp.preprocessing_config_json or "{}")
        model_config = json.loads(exp.model_config_json or "{}")

        gen_res = self.code_gen_agent.run(
            {
                "experiment_id": exp.id,
                "problem_statement": exp.problem_statement,
                "problem_understanding": understanding,
                "selected_dataset": dataset_meta,
                "preprocessing_config": prep_config,
                "model_config": model_config,
            }
        )
        code = gen_res.get("generated_code", "")

        val_res = self.code_val_agent.run({"generated_code": code})
        validation = val_res.get("validation_result", {})

        new_status = (
            ExperimentStatus.READY_TO_RUN.value
            if validation.get("is_valid")
            else ExperimentStatus.FAILED.value
        )
        self.repo.update_experiment(
            experiment_id=experiment_id,
            status=new_status,
            current_step=WorkflowStep.CODE_VALIDATION.value,
            last_validation_json=json.dumps(validation),
        )

        return {"generated_code": code, "validation_result": validation}

    # ─────────────────────────────────────────────────────────────────────────
    # Stage 5: Safe Code Execution
    # ─────────────────────────────────────────────────────────────────────────

    def execute_experiment_code(
        self, experiment_id: str, code: str, timeout: int = 120
    ) -> Dict[str, Any]:
        """
        Execute the generated code safely using CodeExecutionService.
        Returns structured execution result with real metrics.
        """
        from services.code_execution_service import CodeExecutionService

        exp = self.repo.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")

        self.repo.update_experiment(
            experiment_id=experiment_id,
            status=ExperimentStatus.EXECUTING_CODE.value,
            current_step=WorkflowStep.EXECUTION.value,
        )

        service = CodeExecutionService()
        result = service.execute_python_code(code, timeout=timeout, experiment_id=experiment_id)

        # Build results dict for DB
        execution_results = {
            "success": result.success,
            "stdout": result.stdout[-3000:] if result.stdout else "",
            "stderr": result.stderr[-1000:] if result.stderr else "",
            "exit_code": result.exit_code,
            "duration_seconds": result.duration_seconds,
            "error_type": result.error_type,
            "repair_attempts": result.repair_attempts,
            "metrics": result.metrics,
            "results_json": result.results_json,
            "summary": result.results_json.get("summary", ""),
        }

        new_status = (
            ExperimentStatus.COMPLETED.value
            if result.success
            else ExperimentStatus.FAILED.value
        )
        self.repo.update_experiment(
            experiment_id=experiment_id,
            status=new_status,
            current_step=WorkflowStep.EVALUATION.value,
            execution_results_json=json.dumps(execution_results),
        )

        return {
            "success": result.success,
            "execution_result": execution_results,
            "metrics": result.metrics,
            "summary": service.format_execution_summary(result),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Complete Autonomous Experiment Pipeline
    # ─────────────────────────────────────────────────────────────────────────

    def execute_complete_experiment(
        self,
        experiment_id: str,
        problem_statement: str,
        timeout: int = 120,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """
        End-to-end autonomous experiment execution:
        1. Problem understanding
        2. Branch: dataset-free → generate code directly
                   dataset-required → dataset search → analysis → code gen
        3. Execute code safely
        4. Return results
        """
        def _notify(stage: str, message: str):
            if progress_callback:
                progress_callback(stage, message)
            logger.info(f"[{stage}] {message}")

        _notify("understanding", "Analyzing problem statement...")
        formulation = self.process_problem_formulation(experiment_id)
        understanding = formulation["problem_understanding"]

        dataset_required = understanding.get("dataset_required", True)
        experiment_category = understanding.get("experiment_category", "tabular_ml")

        _notify("planning", f"Category: {experiment_category}. Dataset required: {dataset_required}")

        if not dataset_required:
            # Dataset-free path
            _notify("code_generation", "Generating problem-specific code (no dataset needed)...")
            code_result = self.generate_code_without_dataset(experiment_id)
            generated_code = code_result.get("generated_code", "")

        else:
            # Dataset-required path: search → analyze → generate code
            _notify("dataset_search", "Searching for suitable datasets...")
            dataset_result = self.search_and_rank_datasets(experiment_id)
            datasets = dataset_result.get("datasets", [])

            if datasets:
                # Auto-select top-ranked dataset
                best_dataset = datasets[0]
                _notify("dataset_analysis", f"Analyzing dataset: {best_dataset.get('name', 'Unknown')}")
                self.select_and_analyze_dataset(experiment_id, best_dataset)
            else:
                # Fallback: generate code with default sklearn dataset
                _notify("dataset_fallback", "No datasets found — using built-in sklearn dataset.")
                problem_type = understanding.get("problem_type", "tabular_classification")
                fallback_dataset = {
                    "name": "sklearn built-in dataset",
                    "task_type": problem_type,
                    "dataset_id": "sklearn_builtin",
                    "target_column": "target",
                }
                self.select_and_analyze_dataset(experiment_id, fallback_dataset)

            _notify("code_generation", "Generating experiment code...")
            code_result = self.generate_and_validate_code(experiment_id)
            generated_code = code_result.get("generated_code", "")

        # Execute code
        if generated_code:
            _notify("execution", "Executing code in isolated environment...")
            exec_result = self.execute_experiment_code(experiment_id, generated_code, timeout)
            return {
                "problem_understanding": understanding,
                "generated_code": generated_code,
                "execution_result": exec_result,
                "dataset_required": dataset_required,
            }
        else:
            return {
                "problem_understanding": understanding,
                "generated_code": "",
                "execution_result": {"success": False, "metrics": {}},
                "dataset_required": dataset_required,
                "error": "Code generation failed.",
            }
