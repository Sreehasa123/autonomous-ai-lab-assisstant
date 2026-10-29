"""Upgrade Verification Tests: dataset-free detection, code generation, safe execution, and chat intelligence."""
import pytest
import json
from schemas.experiment_schema import ProblemUnderstanding
from agents.problem_understanding_agent import ProblemUnderstandingAgent
from agents.code_generation_agent import CodeGenerationAgent
from services.code_execution_service import CodeExecutionService
from services.chat_service import ChatService
from database.repository import ExperimentRepository
from orchestration.workflow_orchestrator import WorkflowOrchestrator


def test_dataset_free_detection_gradient_descent():
    """Verify that gradient descent / math optimization does NOT require an external dataset."""
    agent = ProblemUnderstandingAgent()
    res = agent.run({
        "problem_statement": "Implement gradient descent to minimize a mathematical function and visualize the convergence process."
    })
    understanding = res["problem_understanding"]
    assert understanding["dataset_required"] is False
    assert understanding["can_run_without_dataset"] is True
    assert understanding["experiment_category"] == "mathematical"


def test_dataset_free_detection_activation_functions():
    """Verify that comparing activation functions does NOT require an external dataset."""
    agent = ProblemUnderstandingAgent()
    res = agent.run({
        "problem_statement": "Compare different activation functions in a neural network and visualize their outputs."
    })
    understanding = res["problem_understanding"]
    assert understanding["dataset_required"] is False
    assert understanding["can_run_without_dataset"] is True
    assert understanding["experiment_category"] == "comparison"


def test_synthetic_data_detection_kmeans():
    """Verify that K-means clustering demo is detected as synthetic/dataset-free."""
    agent = ProblemUnderstandingAgent()
    res = agent.run({
        "problem_statement": "Demonstrate K-Means clustering algorithm on synthetic 2D data points."
    })
    understanding = res["problem_understanding"]
    assert understanding["dataset_required"] is False
    assert understanding["experiment_category"] == "synthetic_data"


def test_dataset_required_detection_plant_disease():
    """Verify that real-world image/tabular tasks require external datasets."""
    agent = ProblemUnderstandingAgent()
    res = agent.run({
        "problem_statement": "Develop a CNN-based deep learning model for classifying plant leaf diseases. Identify a suitable dataset."
    })
    understanding = res["problem_understanding"]
    assert understanding["dataset_required"] is True
    assert understanding["is_deep_learning"] is True


def test_empty_problem_statement_raises():
    """Verify validation error on empty problem statement."""
    agent = ProblemUnderstandingAgent()
    with pytest.raises(ValueError, match="Problem statement is required"):
        agent.run({"problem_statement": ""})


def test_code_execution_service_runs_and_extracts_json():
    """Verify that CodeExecutionService runs python code in isolated subprocess and parses RESULTS_JSON."""
    service = CodeExecutionService()
    test_code = """
import json
val = 42
results = {"metric_a": 0.95, "status": "success", "val": val}
print("Executing demo script...")
print("RESULTS_JSON:" + json.dumps(results))
"""
    result = service.execute_python_code(test_code, timeout=30)
    assert result.success is True
    assert result.exit_code == 0
    assert result.results_json.get("val") == 42
    assert "metric_a" in result.metrics
    assert result.metrics["metric_a"] == 0.95


def test_code_execution_service_handles_error():
    """Verify that CodeExecutionService captures stderr and classifies errors."""
    service = CodeExecutionService()
    bad_code = "raise RuntimeError('Fatal non-recoverable execution failure')"
    result = service.execute_python_code(bad_code, timeout=15)
    assert result.success is False
    assert result.error_type == "RuntimeError"


def test_chat_service_intent_detection():
    """Verify that ChatService correctly identifies intents without needing an LLM."""
    chat = ChatService()

    # Explain intent
    r_explain = chat.process_message("Explain what is gradient descent")
    assert r_explain["intent"] == "explain"
    assert "gradient descent" in r_explain["response_text"].lower()

    # Compare intent
    r_compare = chat.process_message("Compare different activation functions")
    assert r_compare["intent"] == "compare"

    # Full experiment intent
    r_exp = chat.process_message("Implement gradient descent to minimize a mathematical function and visualize convergence")
    assert r_exp["requires_experiment"] is True


def test_orchestrator_autonomous_gradient_descent_pipeline(tmp_path):
    """End-to-end test of dataset-free gradient descent experiment execution."""
    repo = ExperimentRepository()
    orchestrator = WorkflowOrchestrator(repo=repo)

    exp = repo.create_experiment(
        name="Auto Gradient Descent",
        problem_statement="Implement gradient descent to minimize a mathematical function and visualize the convergence process."
    )

    result = orchestrator.execute_complete_experiment(
        experiment_id=exp.id,
        problem_statement=exp.problem_statement,
        timeout=60
    )

    assert result["dataset_required"] is False
    assert len(result["generated_code"]) > 50
    exec_res = result["execution_result"]
    assert exec_res["success"] is True
    assert "optimal_x" in exec_res["metrics"]
    assert "final_loss" in exec_res["metrics"]

    # Verify DB persistence
    saved_exp = repo.get_experiment(exp.id)
    assert saved_exp.status == "completed"
    assert saved_exp.execution_results_json is not None
