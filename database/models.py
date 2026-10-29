"""SQLAlchemy Models for Persistent AI Lab Assistant."""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from database.connection import Base

def generate_uuid() -> str:
    return str(uuid.uuid4())

def utc_now():
    return datetime.now(timezone.utc)

class Project(Base):
    __tablename__ = "projects"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    description = Column(Text, default="")
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    experiments = relationship("Experiment", back_populates="project", cascade="all, delete-orphan")

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=True)
    name = Column(String(255), nullable=False)
    problem_statement = Column(Text, nullable=False)
    status = Column(String(50), default="draft", nullable=False)
    current_version = Column(Integer, default=1, nullable=False)
    current_step = Column(String(100), default="problem_formulation", nullable=False)
    
    # Serialized structured states (JSON)
    workflow_state_json = Column(Text, default="{}")
    problem_understanding_json = Column(Text, default="{}")
    experiment_plan_json = Column(Text, default="{}")
    selected_dataset_json = Column(Text, default="{}")
    dataset_analysis_json = Column(Text, default="{}")
    preprocessing_config_json = Column(Text, default="{}")
    model_config_json = Column(Text, default="{}")
    last_validation_json = Column(Text, default="{}")
    execution_results_json = Column(Text, default="{}")
    notes = Column(Text, default="")
    
    created_at = Column(DateTime, default=utc_now)
    updated_at = Column(DateTime, default=utc_now, onupdate=utc_now)

    # Relationships
    project = relationship("Project", back_populates="experiments")
    versions = relationship("ExperimentVersion", back_populates="experiment", cascade="all, delete-orphan", order_by="ExperimentVersion.version_number", lazy="selectin")
    runs = relationship("ExperimentRun", back_populates="experiment", cascade="all, delete-orphan", order_by="ExperimentRun.run_number", lazy="selectin")
    artifacts = relationship("ArtifactRecord", back_populates="experiment", cascade="all, delete-orphan", lazy="selectin")
    reports = relationship("ReportRecord", back_populates="experiment", cascade="all, delete-orphan", lazy="selectin")

class ExperimentVersion(Base):
    __tablename__ = "experiment_versions"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=False)
    version_number = Column(Integer, nullable=False)
    change_summary = Column(String(500), default="Initial configuration")
    config_snapshot_json = Column(Text, default="{}")
    created_at = Column(DateTime, default=utc_now)

    experiment = relationship("Experiment", back_populates="versions")

class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=False)
    version_number = Column(Integer, default=1, nullable=False)
    run_number = Column(Integer, default=1, nullable=False)
    status = Column(String(50), default="pending", nullable=False)  # pending, running, completed, failed
    execution_mode = Column(String(50), default="local", nullable=False)  # local, colab, remote
    created_at = Column(DateTime, default=utc_now)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, default=0.0)
    
    generated_code = Column(Text, default="")
    execution_logs = Column(Text, default="")
    metrics_json = Column(Text, default="{}")
    error_analysis_json = Column(Text, default="{}")
    model_artifact_path = Column(String(500), nullable=True)
    is_external = Column(Boolean, default=False)

    experiment = relationship("Experiment", back_populates="runs")

class ArtifactRecord(Base):
    __tablename__ = "artifact_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=False)
    run_id = Column(String(36), nullable=True)
    artifact_type = Column(String(50), nullable=False)  # model, plot, notebook, report, dataset, log
    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, default=0)
    checksum = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=utc_now)

    experiment = relationship("Experiment", back_populates="artifacts")

class ReportRecord(Base):
    __tablename__ = "report_records"

    id = Column(String(36), primary_key=True, default=generate_uuid)
    experiment_id = Column(String(36), ForeignKey("experiments.id"), nullable=False)
    title = Column(String(255), nullable=False)
    report_type = Column(String(20), default="markdown")  # markdown, html, json, pdf
    file_path = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=utc_now)

    experiment = relationship("Experiment", back_populates="reports")
