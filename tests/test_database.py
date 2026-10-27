"""Database persistence and CRUD tests."""
import os
import json
import pytest
from database.connection import Base, engine, SessionLocal
from database.models import Experiment, ExperimentVersion, ExperimentRun
from database.repository import ExperimentRepository

@pytest.fixture
def repo():
    Base.metadata.create_all(bind=engine)
    return ExperimentRepository()

def test_database_experiment_lifecycle(repo):
    # 1. Create
    exp = repo.create_experiment(
        name="Test Cancer Diagnostic",
        problem_statement="Classify tumors as benign or malignant."
    )
    assert exp.id is not None
    assert exp.name == "Test Cancer Diagnostic"
    assert exp.current_version == 1
    assert exp.status == "draft"

    # 2. Retrieve
    retrieved = repo.get_experiment(exp.id)
    assert retrieved is not None
    assert retrieved.name == exp.name

    # 3. Update
    repo.update_experiment(
        exp.id,
        status="ready_to_run",
        preprocessing_config_json=json.dumps({"scale_numeric": "standard"}),
        model_config_json=json.dumps({"model_name": "RandomForestClassifier"})
    )
    updated = repo.get_experiment(exp.id)
    assert updated.status == "ready_to_run"
    assert "standard" in updated.preprocessing_config_json

    # 4. Versioning
    new_ver = repo.create_new_version(
        exp.id,
        change_summary="Upgraded to gradient boosting",
        config_snapshot={"model": "GradientBoostingClassifier"}
    )
    assert new_ver.version_number == 2
    reloaded = repo.get_experiment(exp.id)
    assert reloaded.current_version == 2
    assert len(reloaded.versions) == 2

    # 5. Run Creation & Completion
    run = repo.create_run(exp.id, version_number=2, execution_mode="local")
    assert run.run_number == 1
    assert run.status == "running"

    completed_run = repo.complete_run(
        run_id=run.id,
        status="completed",
        duration_seconds=1.25,
        metrics={"accuracy": 0.965, "f1_score": 0.963},
        logs="Training finished smoothly."
    )
    assert completed_run.status == "completed"
    assert completed_run.duration_seconds == 1.25

    # 6. Clean up
    repo.delete_experiment(exp.id)
    assert repo.get_experiment(exp.id) is None
