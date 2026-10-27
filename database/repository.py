"""Database Repository for Autonomous AI Lab Assistant."""
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc

from database.models import Experiment, ExperimentVersion, ExperimentRun, ArtifactRecord, ReportRecord, Project
from database.connection import SessionLocal

logger = logging.getLogger(__name__)

class ExperimentRepository:
    """Repository handling CRUD operations and queries for experiments."""

    def __init__(self, db_session: Optional[Session] = None):
        self._session = db_session

    def get_session(self) -> Session:
        if self._session is not None:
            return self._session
        return SessionLocal()

    def create_experiment(
        self,
        name: str,
        problem_statement: str,
        project_id: Optional[str] = None
    ) -> Experiment:
        session = self.get_session()
        try:
            exp = Experiment(
                name=name,
                problem_statement=problem_statement,
                project_id=project_id,
                status="draft",
                current_version=1,
                current_step="problem_formulation"
            )
            session.add(exp)
            session.commit()
            session.refresh(exp)

            # Create initial version 1
            initial_version = ExperimentVersion(
                experiment_id=exp.id,
                version_number=1,
                change_summary="Initial experiment creation",
                config_snapshot_json=json.dumps({
                    "name": name,
                    "problem_statement": problem_statement
                })
            )
            session.add(initial_version)
            session.commit()
            session.refresh(exp)
            return exp
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating experiment: {e}")
            raise

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        session = self.get_session()
        return session.query(Experiment).filter(Experiment.id == experiment_id).first()

    def list_experiments(
        self,
        status: Optional[str] = None,
        search_query: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Experiment]:
        session = self.get_session()
        query = session.query(Experiment)
        if status and status != "all":
            query = query.filter(Experiment.status == status)
        if search_query:
            query = query.filter(
                Experiment.name.ilike(f"%{search_query}%") |
                Experiment.problem_statement.ilike(f"%{search_query}%")
            )
        query = query.order_by(desc(Experiment.updated_at))
        return query.offset(offset).limit(limit).all()

    def count_experiments_by_status(self) -> Dict[str, int]:
        session = self.get_session()
        experiments = session.query(Experiment.status).all()
        counts = {
            "total": len(experiments),
            "draft": 0,
            "running": 0,
            "completed": 0,
            "failed": 0,
            "awaiting_approval": 0
        }
        for (status,) in experiments:
            if status in counts:
                counts[status] += 1
            elif status == "ready_to_run":
                counts["awaiting_approval"] += 1
        return counts

    def update_experiment(
        self,
        experiment_id: str,
        **kwargs
    ) -> Optional[Experiment]:
        session = self.get_session()
        try:
            exp = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if not exp:
                return None
            for key, val in kwargs.items():
                if hasattr(exp, key):
                    if isinstance(val, (dict, list)):
                        setattr(exp, key, json.dumps(val))
                    else:
                        setattr(exp, key, val)
            exp.updated_at = datetime.now(timezone.utc)
            session.commit()
            session.refresh(exp)
            return exp
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating experiment {experiment_id}: {e}")
            raise

    def create_new_version(
        self,
        experiment_id: str,
        change_summary: str,
        config_snapshot: Dict[str, Any]
    ) -> ExperimentVersion:
        session = self.get_session()
        try:
            exp = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if not exp:
                raise ValueError(f"Experiment {experiment_id} not found")

            new_version_num = exp.current_version + 1
            version = ExperimentVersion(
                experiment_id=experiment_id,
                version_number=new_version_num,
                change_summary=change_summary,
                config_snapshot_json=json.dumps(config_snapshot)
            )
            exp.current_version = new_version_num
            exp.updated_at = datetime.now(timezone.utc)
            exp.versions.append(version)
            session.add(version)
            session.commit()
            session.refresh(exp)
            session.refresh(version)
            return version
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating version for {experiment_id}: {e}")
            raise

    def create_run(
        self,
        experiment_id: str,
        version_number: int,
        execution_mode: str = "local"
    ) -> ExperimentRun:
        session = self.get_session()
        try:
            prev_runs = session.query(ExperimentRun).filter(ExperimentRun.experiment_id == experiment_id).all()
            run_num = len(prev_runs) + 1
            run = ExperimentRun(
                experiment_id=experiment_id,
                version_number=version_number,
                run_number=run_num,
                status="running",
                execution_mode=execution_mode,
                is_external=(execution_mode == "colab")
            )
            exp = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if exp:
                exp.runs.append(run)
            session.add(run)
            session.commit()
            if exp:
                session.refresh(exp)
            session.refresh(run)
            return run
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating run: {e}")
            raise

    def complete_run(
        self,
        run_id: str,
        status: str,
        duration_seconds: float,
        metrics: Dict[str, Any],
        logs: str = "",
        error_analysis: Optional[Dict[str, Any]] = None,
        model_artifact_path: Optional[str] = None
    ) -> Optional[ExperimentRun]:
        session = self.get_session()
        try:
            run = session.query(ExperimentRun).filter(ExperimentRun.id == run_id).first()
            if not run:
                return None
            run.status = status
            run.completed_at = datetime.now(timezone.utc)
            run.duration_seconds = duration_seconds
            run.metrics_json = json.dumps(metrics)
            run.execution_logs = logs
            if error_analysis:
                run.error_analysis_json = json.dumps(error_analysis)
            if model_artifact_path:
                run.model_artifact_path = model_artifact_path
            
            # Also update parent experiment status
            exp = session.query(Experiment).filter(Experiment.id == run.experiment_id).first()
            if exp:
                exp.status = status
                exp.updated_at = datetime.now(timezone.utc)

            session.commit()
            session.refresh(run)
            return run
        except Exception as e:
            session.rollback()
            logger.error(f"Error completing run {run_id}: {e}")
            raise

    def add_artifact(
        self,
        experiment_id: str,
        artifact_type: str,
        file_name: str,
        file_path: str,
        file_size_bytes: int = 0,
        run_id: Optional[str] = None
    ) -> ArtifactRecord:
        session = self.get_session()
        try:
            art = ArtifactRecord(
                experiment_id=experiment_id,
                run_id=run_id,
                artifact_type=artifact_type,
                file_name=file_name,
                file_path=file_path,
                file_size_bytes=file_size_bytes
            )
            exp = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if exp:
                exp.artifacts.append(art)
            session.add(art)
            session.commit()
            if exp:
                session.refresh(exp)
            session.refresh(art)
            return art
        except Exception as e:
            session.rollback()
            logger.error(f"Error adding artifact: {e}")
            raise

    def add_report(
        self,
        experiment_id: str,
        title: str,
        report_type: str,
        file_path: str
    ) -> ReportRecord:
        session = self.get_session()
        try:
            rep = ReportRecord(
                experiment_id=experiment_id,
                title=title,
                report_type=report_type,
                file_path=file_path
            )
            exp = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if exp:
                exp.reports.append(rep)
            session.add(rep)
            session.commit()
            if exp:
                session.refresh(exp)
            session.refresh(rep)
            return rep
        except Exception as e:
            session.rollback()
            logger.error(f"Error saving report: {e}")
            raise

    def duplicate_experiment(self, experiment_id: str, new_name: Optional[str] = None) -> Optional[Experiment]:
        session = self.get_session()
        try:
            original = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if not original:
                return None
            dup = Experiment(
                name=new_name or f"Copy of {original.name}",
                problem_statement=original.problem_statement,
                status="draft",
                current_version=1,
                current_step=original.current_step,
                workflow_state_json=original.workflow_state_json,
                problem_understanding_json=original.problem_understanding_json,
                experiment_plan_json=original.experiment_plan_json,
                selected_dataset_json=original.selected_dataset_json,
                dataset_analysis_json=original.dataset_analysis_json,
                preprocessing_config_json=original.preprocessing_config_json,
                model_config_json=original.model_config_json,
                notes=f"Duplicated from {original.name}"
            )
            session.add(dup)
            session.commit()
            session.refresh(dup)
            
            # Add initial version for duplicated experiment
            ver = ExperimentVersion(
                experiment_id=dup.id,
                version_number=1,
                change_summary=f"Duplicated from {original.name}",
                config_snapshot_json=json.dumps({"source_experiment_id": original.id})
            )
            session.add(ver)
            session.commit()
            return dup
        except Exception as e:
            session.rollback()
            logger.error(f"Error duplicating experiment {experiment_id}: {e}")
            raise

    def delete_experiment(self, experiment_id: str) -> bool:
        session = self.get_session()
        try:
            exp = session.query(Experiment).filter(Experiment.id == experiment_id).first()
            if not exp:
                return False
            session.delete(exp)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting experiment {experiment_id}: {e}")
            raise
