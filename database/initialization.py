"""Database Initialization and Migration safety."""
import logging
from database.connection import engine, Base
from database.models import Project, Experiment, ExperimentVersion, ExperimentRun, ArtifactRecord, ReportRecord

logger = logging.getLogger(__name__)

def init_db():
    """Create all database tables safely without overwriting existing data."""
    try:
        Base.metadata.create_all(bind=engine)
        # Safe lightweight schema migration for SQLite
        with engine.connect() as conn:
            try:
                conn.execute(
                    __import__("sqlalchemy").text("ALTER TABLE experiments ADD COLUMN execution_results_json TEXT DEFAULT '{}'")
                )
                conn.commit()
            except Exception:
                pass  # column already exists
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error during database initialization: {e}")
        raise
