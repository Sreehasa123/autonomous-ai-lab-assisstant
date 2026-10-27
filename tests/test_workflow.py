"""Workflow Orchestration and State Transition Tests."""
import pytest
from database.repository import ExperimentRepository
from orchestration.workflow_orchestrator import WorkflowOrchestrator
from orchestration.workflow_state import WorkflowStateMachine, WorkflowStep
from schemas.experiment_schema import ExperimentStatus

def test_workflow_state_machine_and_invalidation():
    sm = WorkflowStateMachine()
    assert sm.current_step == WorkflowStep.PROBLEM_UNDERSTANDING
    
    sm.mark_completed(WorkflowStep.PROBLEM_UNDERSTANDING, {"data": 1})
    sm.mark_completed(WorkflowStep.DATASET_SELECTION, {"dataset": "iris"})
    sm.mark_completed(WorkflowStep.PREPROCESSING, {"scaled": True})
    
    assert sm.is_completed(WorkflowStep.PROBLEM_UNDERSTANDING)
    assert sm.is_completed(WorkflowStep.PREPROCESSING)

    # Invalidate from dataset selection
    sm.invalidate_from(WorkflowStep.DATASET_SELECTION)
    assert sm.is_completed(WorkflowStep.PROBLEM_UNDERSTANDING)
    assert not sm.is_completed(WorkflowStep.DATASET_SELECTION)
    assert not sm.is_completed(WorkflowStep.PREPROCESSING)

def test_orchestrator_end_to_end_pipeline():
    repo = ExperimentRepository()
    orchestrator = WorkflowOrchestrator(repo=repo)

    # 1. Problem formulation
    exp = repo.create_experiment("E2E Test Experiment", "Classify iris flower plant types based on sepal and petal dimensions.")
    res_formulation = orchestrator.process_problem_formulation(exp.id)
    assert res_formulation["problem_understanding"]["problem_type"] == "tabular_classification"

    # 2. Select dataset
    uci_ds = {
        "dataset_id": "uci_iris",
        "name": "UCI: Iris Flower Dataset",
        "source_provider": "uci",
        "task_type": "classification",
        "target_column": "target"
    }
    res_select = orchestrator.select_and_analyze_dataset(exp.id, uci_ds)
    assert res_select["analysis_result"]["shape"][0] > 0
    assert len(res_select["recommended_models"]) > 0

    # 3. Generate & Validate Code
    res_code = orchestrator.generate_and_validate_code(exp.id)
    assert res_code["validation_result"]["is_valid"] is True
    assert "RandomForestClassifier" in res_code["generated_code"]

    # 4. Clean up
    repo.delete_experiment(exp.id)
