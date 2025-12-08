"""
Database initialization script.
Creates all tables in the SQLite database.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_settings
from app.models.base import init_db, create_tables, get_engine
from sqlalchemy import inspect
import os


def check_existing_database():
    """Check if database already exists."""
    settings = load_settings()
    db_path = Path(settings.sqlite_db_path)
    return db_path.exists()


def drop_all_tables():
    """Drop all existing tables."""
    from app.models.base import Base
    from app.models.conversation import Conversation
    from app.models.message import Message
    from app.models.workflow import WorkflowState
    
    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    print("✓ All tables dropped")


def verify_tables():
    """Verify that all tables were created."""
    engine = get_engine()
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    
    expected_tables = ['conversations', 'messages', 'workflow_states']
    missing_tables = [t for t in expected_tables if t not in tables]
    
    if missing_tables:
        print(f"✗ Missing tables: {', '.join(missing_tables)}")
        return False
    
    print(f"✓ All tables created: {', '.join(tables)}")
    return True


def main():
    """Main initialization logic."""
    print("=" * 60)
    print("Database Initialization")
    print("=" * 60)
    
    # Load settings
    try:
        settings = load_settings()
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        sys.exit(1)
    
    # Check if database exists
    db_exists = check_existing_database()
    
    if db_exists:
        print(f"⚠ Database already exists at: {settings.sqlite_db_path}")
        response = input("Drop and recreate all tables? (y/n): ").strip().lower()
        
        if response != 'y':
            print("Aborted.")
            sys.exit(0)
        
        # Drop existing tables
        try:
            init_db()
            drop_all_tables()
        except Exception as e:
            print(f"✗ Error dropping tables: {e}")
            sys.exit(1)
    else:
        print(f"Creating new database at: {settings.sqlite_db_path}")
    
    # Initialize database
    try:
        init_db()
        print("✓ Database engine initialized")
    except Exception as e:
        print(f"✗ Error initializing database: {e}")
        sys.exit(1)
    
    # Create tables
    try:
        create_tables()
    except Exception as e:
        print(f"✗ Error creating tables: {e}")
        sys.exit(1)
    
    # Verify tables
    if not verify_tables():
        sys.exit(1)
    
    print("=" * 60)
    print("✓ Database initialization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()