"""Application Settings and Configuration."""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field

# Base root directory of the application
BASE_DIR = Path(__file__).resolve().parent.parent

def load_dotenv_fallback(dotenv_path: Path):
    """Simple parser for .env file if python-dotenv is not installed."""
    if not dotenv_path.exists():
        return
    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass

# Attempt to load .env
load_dotenv_fallback(BASE_DIR / ".env")

class Settings(BaseModel):
    """Central configuration for Autonomous AI Lab Assistant."""
    
    app_name: str = "Autonomous AI Lab Assistant"
    version: str = "1.0.0"
    debug: bool = Field(default=False)
    
    # Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    datasets_dir: Path = BASE_DIR / "data" / "datasets"
    cache_dir: Path = BASE_DIR / "data" / "cache"
    
    artifacts_dir: Path = BASE_DIR / "artifacts"
    models_dir: Path = BASE_DIR / "artifacts" / "models"
    plots_dir: Path = BASE_DIR / "artifacts" / "plots"
    reports_dir: Path = BASE_DIR / "artifacts" / "reports"
    notebooks_dir: Path = BASE_DIR / "artifacts" / "notebooks"
    logs_dir: Path = BASE_DIR / "artifacts" / "logs"
    templates_dir: Path = BASE_DIR / "templates" / "reports"
    
    # Database configuration
    database_url: str = Field(
        default_factory=lambda: os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'lab_assistant.db'}")
    )
    
    # Execution & Limits
    default_execution_mode: str = "local"  # local or colab
    default_framework: str = "scikit-learn"  # scikit-learn, pytorch
    max_preview_rows: int = 100
    max_dataset_download_mb: int = 250
    execution_timeout_seconds: int = 300
    random_seed: int = 42
    
    # Optional LLM integration
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "none"))  # none, openai, gemini, anthropic
    llm_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("LLM_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"))
    llm_model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gemini-1.5-flash"))
    
    # Optional Dataset Providers Credentials
    hf_token: Optional[str] = Field(default_factory=lambda: os.getenv("HF_TOKEN"))
    kaggle_username: Optional[str] = Field(default_factory=lambda: os.getenv("KAGGLE_USERNAME"))
    kaggle_key: Optional[str] = Field(default_factory=lambda: os.getenv("KAGGLE_KEY"))
    openml_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENML_API_KEY"))
    
    def ensure_directories(self) -> None:
        """Create required runtime directories safely."""
        dirs = [
            self.data_dir,
            self.datasets_dir,
            self.cache_dir,
            self.artifacts_dir,
            self.models_dir,
            self.plots_dir,
            self.reports_dir,
            self.notebooks_dir,
            self.logs_dir,
            self.templates_dir
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

# Singleton settings instance
settings = Settings()
settings.ensure_directories()
