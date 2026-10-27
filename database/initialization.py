"""Database Initialization and Migration safety."""
import logging
from database.connection import engine, Base
from database.models import Project, Experiment, ExperimentVersion, ExperimentRun, ArtifactRecord, ReportRecord

logger = logging.getLogger(__name__)

def init_db():
    """Create all database tables safely without overwriting existing data."""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise
