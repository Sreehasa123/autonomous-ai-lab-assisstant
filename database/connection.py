"""Database Connection and Session Management."""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from config.settings import settings

Base = declarative_base()

# SQLite engine with check_same_thread=False for multi-threaded Streamlit
connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    echo=False,
    pool_pre_ping=True
)

SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine, expire_on_commit=False))

def get_db():
    """Dependency / context generator for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
