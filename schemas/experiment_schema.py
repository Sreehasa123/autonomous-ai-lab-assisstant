"""Experiment Schemas and Definitions — Upgraded with dataset-free support."""
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ExperimentStatus(str, Enum):
    """Workflow states for an experiment."""
    DRAFT = "draft"
    PLANNING = "planning"
    SEARCHING_DATASETS = "searching_datasets"
    DATASET_SELECTED = "dataset_selected"
    ANALYZING_DATA = "analyzing_data"
    PREPROCESSING = "preprocessing"
    AWAITING_APPROVAL = "awaiting_approval"
    CODE_GENERATION = "code_generation"
    VALIDATION = "validation"
    READY_TO_RUN = "ready_to_run"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    AWAITING_EXTERNAL = "awaiting_external_execution"
    # New statuses for upgrade
    DATASET_NOT_REQUIRED = "dataset_not_required"
    GENERATING_CODE = "generating_code"
    EXECUTING_CODE = "executing_code"
    REPAIRING_CODE = "repairing_code"


class ProblemUnderstanding(BaseModel):
    """Structured understanding of the user's natural language problem."""
    problem_type: str = "tabular_classification"
    objective: str = ""
    expected_input: str = ""
    expected_output: str = ""
    likely_data_format: str = "csv"
    target_variable: Optional[str] = None
    candidate_metrics: List[str] = Field(default_factory=list)
    possible_model_families: List[str] = Field(default_factory=list)
    preprocessing_requirements: List[str] = Field(default_factory=list)
    potential_risks: List[str] = Field(default_factory=list)
    is_deep_learning: bool = False
    explanation: str = ""
    # New fields for upgrade — dataset-free support
    dataset_required: bool = True
    experiment_category: str = "tabular_ml"   # tabular_ml, deep_learning, mathematical, synthetic_data, nlp, time_series, computer_vision, comparison
    workflow_stages: List[str] = Field(default_factory=list)
    can_run_without_dataset: bool = False


class ExperimentPlan(BaseModel):
    """Comprehensive plan generated for the experiment."""
    objectives: List[str] = Field(default_factory=list)
    dataset_requirements: Dict[str, Any] = Field(default_factory=dict)
    search_strategy: List[str] = Field(default_factory=list)
    preprocessing_plan: List[str] = Field(default_factory=list)
    candidate_models: List[Dict[str, Any]] = Field(default_factory=list)
    evaluation_methodology: str = ""
    resource_considerations: str = ""
    expected_deliverables: List[str] = Field(default_factory=list)
    potential_challenges: List[str] = Field(default_factory=list)


class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing."""
    handle_missing: str = "median_mode"
    scale_numeric: str = "standard"
    encode_categorical: str = "onehot"
    remove_duplicates: bool = True
    handle_imbalance: str = "none"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    feature_selection: str = "none"


class ModelConfig(BaseModel):
    """Configuration for model selection and training."""
    model_name: str = "RandomForestClassifier"
    model_family: str = "ensemble"
    framework: str = "scikit-learn"
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    is_baseline: bool = True
    explanation: str = ""


class CodeValidationResult(BaseModel):
    """Validation report for generated code."""
    is_valid: bool = False
    syntax_valid: bool = False
    imports_valid: bool = False
    schema_valid: bool = False
    smoke_test_passed: bool = False
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    repair_attempts: int = 0
    final_code: str = ""


class ExecutionResult(BaseModel):
    """Result from safe code execution engine."""
    success: bool = False
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    duration_seconds: float = 0.0
    artifacts: List[str] = Field(default_factory=list)
    error_type: str = ""
    repair_attempts: int = 0
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ChatIntent(str, Enum):
    """Detected intent in a chat message."""
    EXPLAIN = "explain"
    GENERATE_CODE = "generate_code"
    EXECUTE = "execute"
    IMPROVE = "improve"
    COMPARE = "compare"
    SAVE = "save"
    RESUME = "resume"
    NEW_EXPERIMENT = "new_experiment"
    UNKNOWN = "unknown"


class ChatResponse(BaseModel):
    """Response from the chat service."""
    response_text: str = ""
    intent: ChatIntent = ChatIntent.UNKNOWN
    action: str = ""
    requires_experiment: bool = False
    experiment_update: Dict[str, Any] = Field(default_factory=dict)
    generated_code: str = ""
    suggested_followups: List[str] = Field(default_factory=list)
