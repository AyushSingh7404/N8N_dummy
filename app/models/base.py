"""
SQLAlchemy base model and database session management.
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
from config import get_settings

# Base class for all models
Base = declarative_base()

# Database engine (initialized on first import)
_engine = None
_SessionLocal = None


def init_db():
    """Initialize database engine and session maker."""
    global _engine, _SessionLocal
    
    settings = get_settings()
    
    # Create SQLite engine
    _engine = create_engine(
        f"sqlite:///{settings.sqlite_db_path}",
        connect_args={"check_same_thread": False},  # Needed for SQLite
        echo=settings.flask_debug  # Log SQL queries in debug mode
    )
    
    # Create session maker
    _SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=_engine
    )
    
    return _engine


def create_tables():
    """Create all tables in the database."""
    from app.models.conversation import Conversation
    from app.models.message import Message
    from app.models.workflow import WorkflowState
    
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully")


def get_engine():
    """Get database engine, initializing if needed."""
    global _engine
    if _engine is None:
        init_db()
    return _engine


def get_session_maker():
    """Get session maker, initializing if needed."""
    global _SessionLocal
    if _SessionLocal is None:
        init_db()
    return _SessionLocal


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Get database session context manager.
    
    Usage:
        with get_db() as db:
            db.query(...)
    
    Yields:
        Session: Database session
    """
    SessionLocal = get_session_maker()
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_dependency():
    """
    FastAPI dependency for database session.
    
    Usage:
        @app.get("/")
        def route(db: Session = Depends(get_db_dependency)):
            ...
    """
    SessionLocal = get_session_maker()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()