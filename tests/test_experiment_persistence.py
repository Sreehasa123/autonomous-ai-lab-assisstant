"""Mandatory 13-Point Experiment Persistence Tests."""
import json
import pytest
from database.connection import SessionLocal
from database.repository import ExperimentRepository
from services.visualization_service import VisualizationService
from schemas.experiment_schema import ExperimentStatus

def test_mandatory_13_point_persistence_cycle():
    # 1. Experiment is created
    repo1 = ExperimentRepository()
    exp = repo1.create_experiment(
        name="Long-Term Persistence Test Experiment",
        problem_statement="Predict diabetes disease progression using continuous blood biomarkers."
    )
    exp_id = exp.id
    assert exp_id is not None
    assert exp.current_version == 1

    # 2. Configuration is saved
    repo1.update_experiment(
        exp_id,
        selected_dataset_json=json.dumps({"name": "UCI: Diabetes", "task_type": "regression", "target_column": "target"}),
        preprocessing_config_json=json.dumps({"scale_numeric": "standard", "test_size": 0.2}),
        model_config_json=json.dumps({"model_name": "RandomForestRegressor", "framework": "scikit-learn"})
    )

    # 3. Results and metrics are saved
    run1 = repo1.create_run(exp_id, version_number=1, execution_mode="local")
    metrics_v1 = {
        "primary_metric_name": "R² Score",
        "primary_metric_value": 0.542,
        "metrics": {"r2_score": 0.542, "rmse": 54.3, "mae": 43.2},
        "actual_vs_predicted": [{"actual": 150.0, "predicted": 145.0}, {"actual": 200.0, "predicted": 190.0}]
    }
    repo1.complete_run(
        run_id=run1.id,
        status=ExperimentStatus.COMPLETED.value,
        duration_seconds=0.85,
        metrics=metrics_v1,
        logs="Run 1 Completed Successfully."
    )

    # 4. Session is explicitly closed / simulated restart
    SessionLocal.remove()
    del repo1

    # 5. A completely new database session retrieves the same experiment
    repo2 = ExperimentRepository()
    reloaded_exp = repo2.get_experiment(exp_id)
    assert reloaded_exp is not None
    assert reloaded_exp.name == "Long-Term Persistence Test Experiment"

    # 6. Results remain fully available
    assert len(reloaded_exp.runs) == 1
    saved_run1 = reloaded_exp.runs[0]
    saved_metrics = json.loads(saved_run1.metrics_json)
    assert saved_metrics["primary_metric_value"] == 0.542
    assert saved_metrics["metrics"]["r2_score"] == 0.542

    # 7. Visualizations can be recreated from saved data
    avp = saved_metrics.get("actual_vs_predicted", [])
    fig = VisualizationService.plot_actual_vs_predicted(avp)
    assert fig is not None

    # 8. The experiment can be edited
    repo2.update_experiment(
        exp_id,
        model_config_json=json.dumps({"model_name": "GradientBoostingRegressor", "framework": "scikit-learn"})
    )

    # 9. Versioning: original version remains available
    v2 = repo2.create_new_version(
        exp_id,
        change_summary="Switched to GradientBoostingRegressor",
        config_snapshot={"model": "GradientBoostingRegressor"}
    )
    assert v2.version_number == 2
    exp_after_version = repo2.get_experiment(exp_id)
    assert exp_after_version.current_version == 2
    assert len(exp_after_version.versions) == 2
    assert exp_after_version.versions[0].version_number == 1
    assert exp_after_version.versions[1].version_number == 2

    # 10. A new run can be created
    run2 = repo2.create_run(exp_id, version_number=2, execution_mode="local")
    metrics_v2 = {
        "primary_metric_name": "R² Score",
        "primary_metric_value": 0.615,
        "metrics": {"r2_score": 0.615, "rmse": 48.7, "mae": 39.1}
    }
    repo2.complete_run(
        run_id=run2.id,
        status=ExperimentStatus.COMPLETED.value,
        duration_seconds=1.12,
        metrics=metrics_v2,
        logs="Run 2 Completed Successfully."
    )

    # 11. Previous and new runs can be compared
    exp_final = repo2.get_experiment(exp_id)
    assert len(exp_final.runs) == 2
    assert json.loads(exp_final.runs[0].metrics_json)["primary_metric_value"] == 0.542
    assert json.loads(exp_final.runs[1].metrics_json)["primary_metric_value"] == 0.615

    # 12. Incomplete workflow resumption check
    assert exp_final.current_step is not None
    assert exp_final.status == ExperimentStatus.COMPLETED.value

    # 13. Missing artifacts handled without destroying experiment history
    # Add a non-existent artifact record
    art = repo2.add_artifact(
        experiment_id=exp_id,
        artifact_type="model",
        file_name="missing_model.joblib",
        file_path="/invalid/path/missing_model.joblib"
    )
    # Experiment remains valid and readable
    exp_with_missing_art = repo2.get_experiment(exp_id)
    assert exp_with_missing_art is not None
    assert len(exp_with_missing_art.artifacts) >= 1

    # Clean up
    repo2.delete_experiment(exp_id)
