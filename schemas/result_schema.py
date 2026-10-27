"""Result and Evaluation Schemas."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class EvaluationMetrics(BaseModel):
    """Measured evaluation metrics from a real experiment run."""
    task_type: str = "classification"  # classification, regression, clustering, time_series
    primary_metric_name: str = "accuracy"
    primary_metric_value: float = 0.0
    metrics: Dict[str, float] = Field(default_factory=dict)
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    confusion_matrix: Optional[List[List[int]]] = None
    class_labels: Optional[List[str]] = None
    actual_vs_predicted: Optional[List[Dict[str, float]]] = None
    feature_importances: Optional[Dict[str, float]] = None
    training_history: Optional[Dict[str, List[float]]] = None  # epochs, loss, accuracy, val_loss, val_accuracy
    execution_duration_sec: float = 0.0

class ErrorFinding(BaseModel):
    """Diagnostic finding from Error Analysis Agent."""
    issue: str
    severity: str  # low, medium, high
    evidence: str
    explanation: str

class ImprovementRecommendation(BaseModel):
    """Specific actionable improvement suggested for the next experiment version/run."""
    category: str  # hyperparameter, preprocessing, model_architecture, data_augmentation, regularization
    title: str
    action: str
    rationale: str
    estimated_impact: str  # low, moderate, significant
    suggested_parameters: Dict[str, Any] = Field(default_factory=dict)

class ErrorAnalysisReport(BaseModel):
    """Overall report from error analysis and improvement agents."""
    overfitting_assessment: str = "Normal (train/validation gap is within healthy bounds)"
    underfitting_assessment: str = "Low (model captures primary patterns)"
    findings: List[ErrorFinding] = Field(default_factory=list)
    recommendations: List[ImprovementRecommendation] = Field(default_factory=list)
