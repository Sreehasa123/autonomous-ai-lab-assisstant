"""Dataset Schemas and Data Models."""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

class DatasetMetadata(BaseModel):
    """Metadata representing a dataset across different sources."""
    dataset_id: str
    name: str
    description: str = ""
    source_provider: str  # huggingface, openml, kaggle, uci, local_upload, built_in
    url: Optional[str] = None
    download_url: Optional[str] = None
    data_format: str = "csv"  # csv, parquet, json, image_folder, text
    num_rows: Optional[int] = None
    num_features: Optional[int] = None
    target_column: Optional[str] = None
    task_type: Optional[str] = None  # classification, regression, clustering, nlp, cv, time_series
    license: Optional[str] = "Unknown"
    size_mb: Optional[float] = None
    class_distribution: Optional[Dict[str, int]] = None
    is_verified: bool = False
    local_path: Optional[str] = None
    suitability_score: Optional[float] = None  # 0 to 100
    suitability_reason: Optional[str] = None
    historical_benchmarks: List[Dict[str, Any]] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)

class DataQualityFinding(BaseModel):
    """Specific finding from the Data Quality Agent."""
    severity: str = "info"  # info, warning, high_risk, critical
    category: str  # missing_values, duplicates, leakage, imbalance, types, outliers
    description: str
    impact: str
    suggested_action: str

class DatasetAnalysisResult(BaseModel):
    """Statistical and structural analysis of a dataset."""
    shape: List[int] = Field(default_factory=lambda: [0, 0])
    columns: List[str] = Field(default_factory=list)
    dtypes: Dict[str, str] = Field(default_factory=dict)
    sample_records: List[Dict[str, Any]] = Field(default_factory=list)
    missing_counts: Dict[str, int] = Field(default_factory=dict)
    duplicate_rows: int = 0
    numerical_summary: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    categorical_summary: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    target_distribution: Optional[Dict[str, Any]] = None
    potential_leakage_columns: List[str] = Field(default_factory=list)
    constant_columns: List[str] = Field(default_factory=list)
    identifier_columns: List[str] = Field(default_factory=list)
    findings: List[DataQualityFinding] = Field(default_factory=list)
