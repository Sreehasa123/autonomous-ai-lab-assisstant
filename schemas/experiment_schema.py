"""Experiment Schemas and Definitions."""
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

class ProblemUnderstanding(BaseModel):
    """Structured understanding of the user's natural language problem."""
    problem_type: str = "tabular_classification"  # tabular_classification, tabular_regression, image_classification, text_classification, time_series, clustering
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
    handle_missing: str = "median_mode"  # mean, median, drop, none
    scale_numeric: str = "standard"  # standard, minmax, robust, none
    encode_categorical: str = "onehot"  # onehot, ordinal, label, none
    remove_duplicates: bool = True
    handle_imbalance: str = "none"  # class_weight, none
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    feature_selection: str = "none"  # none, variance, mutual_info

class ModelConfig(BaseModel):
    """Configuration for model selection and training."""
    model_name: str = "RandomForestClassifier"
    model_family: str = "ensemble"
    framework: str = "scikit-learn"  # scikit-learn, pytorch
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
